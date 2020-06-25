from mesa import Agent, Model
from mesa.time import RandomActivation
import networkx as nx
from enum import Enum
from typing import Tuple
from collections import deque
import random
import math
import uuid
import pprint
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# in 10th of seconds
SIMULATION_RUN_STEPS = 9000
MAC_ROTATE_STEPS_TOTAL = 9000
RPI_ROTATE_STEPS_MIN = 6000
RPI_ROTATE_STEPS_MAX = 9000
ADVERTISE_MIN = 2
ADVERTISE_MAX = 3 # 2.7 according to spec
ADVERTISE_GALAXY_S10_MIN = 30
ADVERTISE_GALAXY_S10_MAX = 50

STEP_RESOLUTION_MS = 100

SPATIAL_UNIQ_DIST = 3

# number of steps for 1 square map movement
MOVEMENT_STEPS_PER_SQUARE = 10

# track last map step trigger
map_step = 0

# replay data is global
map_movie = list ()
action_list = deque()
movement_list = deque()

class ExposureNotificationTimerTypes(Enum):
    """Timer setup according to Google/Apple specifications or specific Android test"""
    EXPOSURE_NOTIFICATION_TIMER_SPEC = 1
    EXPOSURE_NOTIFICATION_TIMER_ANDROID = 2
    EXPOSURE_NOTIFICATION_TIMER_ANDROID_R = 3
    EXPOSURE_NOTIFICATION_TIMER_ANDROID_R_FIXED = 4
    EXPOSURE_NOTIFICATION_TIMER_SPEC_FIXED = 5


class Map:
    
    """Map with NxN square fields where agent moves around.
       Agent switches position with agent already inhibiting target square."""
    def __init__(self, cluster_size, side_length):
        self._map = [[None for i in range(side_length)] for j in range(side_length)]
        self._cluster_size = cluster_size
        self._side_length = side_length
        self._placed = 0
        self._step = 0


    def place_agent(self, agent_id):
        """Place agent on random free square"""
        assert self._placed < (self._side_length * self._side_length)

        x = y = None
        while x is None and y is None:
            x = random.randint(0, self._side_length - 1)
            y = random.randint(0, self._side_length - 1)

            if self._map[x][y] is None:
                self._map[x][y] = agent_id
                self._placed += 1
            else:
                x = y = None

    def spatially_unique(self, agent1_id, agent2_id):
        """Check if two agents are spatially unique (at least one square apart)"""

        a1x, a1y = self.locate_agent(agent1_id)
        a2x, a2y = self.locate_agent(agent2_id)

        return abs(a1x - a2x) >= SPATIAL_UNIQ_DIST or abs(a1y - a2y) >= SPATIAL_UNIQ_DIST

    def locate_agent(self, agent_id):
        """Locate agent by ID"""

        for x in range(self._side_length):
            for y in range(self._side_length):
                if self._map[x][y] == agent_id:
                    return x, y

        raise ValueError(f"unable to locate agent {agent_id}")

    def __str__(self):
        return pprint.pformat([self._map[i, j] for i in self._map for j in self._map[i]])

    def step(self):
        """Agents move in horizontal or diagonal direction for 1 square every movement steps"""

        global movement_list, map_movie

        has_moved = False
        agent_ids = set([a for r in self._map for a in r if not a is None])
        agent_slice = MOVEMENT_STEPS_PER_SQUARE / self._cluster_size
        for agent_id in agent_ids:
            agent_offset = math.floor(agent_slice * agent_id)
            if (self._step + agent_offset) % MOVEMENT_STEPS_PER_SQUARE == 0:
                x, y = self.locate_agent(agent_id)
                dx, dy = random.randrange(-1, 2), random.randrange(-1, 2)

                if (x + dx) >= len(self._map[0]) or \
                   (y + dy) >= len(self._map):
                   
                   continue

                has_moved = True

                if self._map[x + dx][y + dy] is None:
                    self._map[x][y] = None
                    movement_list += [(self._step, x, y, None)]
                else:
                    source = self._map[x + dx][y + dy]
                    self._map[x][y] = source
                    movement_list += [(self._step, x, y, source)]

                self._map[x + dx][y + dy] = agent_id
                movement_list += [(self._step, x + dx, y + dy, agent_id)]

        if has_moved:
            map_movie += [(self._step, str(self._map))]

        self._step += 1

TimeSeriesDatum = Tuple[int, str, str, int, int]

class TimeSeriesData:
    """Captures all simulation events for post processing"""
    def __init__(self):
        self.data = []
    
    def add(self, datum: TimeSeriesDatum):
        """Add record to time series database"""
        self.data += [datum]

    def trackable_fraction(self, cluster_size, advertising_interval):
        """Follow all trackable devices in collected data and report tracking success"""
        g = nx.DiGraph()
        traversal_lut_mac = {} # lookup rotations with known identifier directly carrying over
        traversal_lut_rpi = {}
        lost_tracks = 0

        temporal_lookahead = deque() # look ahead one advertising_interval to see if this tuple is
                                        # unique at once - at that point it can be considered a unique
                                        # rotation indirectly carrying over

        for step, mac, rpi, x, y in self.data + [(99999999, None, None, -1, -1)]:
            if not (mac, rpi) == (None, None): # skip processing for last rerun to cleanup deque

                if (mac, rpi) not in g.nodes.items():
                    g.add_node((mac, rpi))

                # direct traversals via rpi
                if mac in traversal_lut_mac:

                    prev_rpi = traversal_lut_mac[mac]
                    del traversal_lut_mac[mac]
                    del traversal_lut_rpi[prev_rpi]

                    action_list.append((step, 'traverse_rpi', x, y))
                    g.add_edge((mac, prev_rpi), (mac, rpi))

                # direct traversals via mac
                elif rpi in traversal_lut_rpi:

                    prev_mac = traversal_lut_rpi[rpi]
                    del traversal_lut_mac[prev_mac]
                    del traversal_lut_rpi[rpi]

                    action_list.append((step, 'traverse_mac', x, y))
                    g.add_edge((prev_mac, rpi), (mac, rpi))

                # indirect lookup via temporal rotation uniqueness
                else:
                    temporal_lookahead.append((step, mac, rpi, x, y))

                # save lut for following ref cycles
                traversal_lut_mac[mac] = rpi
                traversal_lut_rpi[rpi] = mac

            shift_n = 0

            # process temporal lookahead for current step
            while len(temporal_lookahead) >= 2 and \
                    (step - temporal_lookahead[0][0]) >= (2 * advertising_interval) and \
                    (step - temporal_lookahead[1][0]) >= advertising_interval:

                prev_action_list_len = len(action_list)

                recover_src = recover_dst = None               
                if (temporal_lookahead[1][0] - temporal_lookahead[0][0]) >= advertising_interval:
                    # temporal uniqueness
                    recover_src = (temporal_lookahead[0][1], temporal_lookahead[0][2])
                    recover_dst = (temporal_lookahead[1][1], temporal_lookahead[1][2])

                    action_list.append((step, 'recover_temporal', x, y))
                    g.add_edge(recover_src, recover_dst)
                else:
                    # spatial uniqueness
                    x0, y0 = tuple(temporal_lookahead[0][3:5])
                    x1, y1 = tuple(temporal_lookahead[1][3:5])

                    if abs(x0-x1) >= SPATIAL_UNIQ_DIST or abs(y0-y1) >= SPATIAL_UNIQ_DIST:
                        recover_src = (temporal_lookahead[0][1], temporal_lookahead[0][2])
                        recover_dst = (temporal_lookahead[1][1], temporal_lookahead[1][2])

                        action_list.append((temporal_lookahead[1][0], 'recover_spatial', x1, y1, x0, y0))
                        g.add_edge(recover_src, recover_dst)

                if not (recover_src is None or recover_dst is None):
                    g.add_edge(recover_src, recover_dst)
                    temporal_lookahead.popleft()
                    temporal_lookahead.popleft()


                # lost track on oldest stale items if not already used in processing
                # (must be too narrow to neighbor)
                while shift_n < len(temporal_lookahead) and \
                    (step - temporal_lookahead[shift_n][0]) >= (2 * advertising_interval):
                    
                    shift_n += 1

                    x, y = tuple(temporal_lookahead[0][3:5])
                    action_list.append((step, 'track_lost', x, y))
                
                for _ in range(shift_n):
                    temporal_lookahead.popleft()
            
                lost_tracks += shift_n

                if prev_action_list_len == len(action_list):
                    break # no action available, break processing
        
        if len(self.data) == 0:
            return 1.0
        else:
            return 1.0 - lost_tracks / len(self.data)


class ExposureNotificationTimers:
    """Timer model for Google/Apple ExposureNotification specs"""
    def __init__(self, timer_type: ExposureNotificationTimerTypes, guid: int, time_series_data: TimeSeriesData, space_map: Map):
        self._time_series_data = time_series_data
        self._guid = guid
        self._step = 0
        self.is_fixed = False
        self._space_map = space_map
        self._rotate_mac()
        self._rotate_rpi()

        {
            ExposureNotificationTimerTypes.EXPOSURE_NOTIFICATION_TIMER_SPEC: ExposureNotificationTimers._gen_EXPOSURE_NOTIFICATION_TIMER_SPEC,
            ExposureNotificationTimerTypes.EXPOSURE_NOTIFICATION_TIMER_ANDROID: ExposureNotificationTimers._gen_EXPOSURE_NOTIFICATION_TIMER_ANDROID,
            ExposureNotificationTimerTypes.EXPOSURE_NOTIFICATION_TIMER_ANDROID_R: ExposureNotificationTimers._gen_EXPOSURE_NOTIFICATION_TIMER_ANDROID_R,
            ExposureNotificationTimerTypes.EXPOSURE_NOTIFICATION_TIMER_ANDROID_R_FIXED: ExposureNotificationTimers._gen_EXPOSURE_NOTIFICATION_TIMER_ANDROID_R_FIXED,
            ExposureNotificationTimerTypes.EXPOSURE_NOTIFICATION_TIMER_SPEC_FIXED: ExposureNotificationTimers._gen_EXPOSURE_NOTIFICATION_TIMER_SPEC_FIXED
        }[timer_type](self)

    def step(self):
        global map_step

        """Execute simulation step"""
        if self._step % self.mac_rotate_step == 0:
            self._rotate_mac()
        if self._step % self.rpi_rotate_step == 0:
            self._rotate_rpi()
        if self._step % self.advertise_step == 0:
            self._advertise()
        
        self._step += 1

        if map_step < self._step:
            self._space_map.step()
            map_step += 1

    def _gen_EXPOSURE_NOTIFICATION_TIMER_SPEC(self):
        """Generate timers according to Google/Apple spec"""
        self.mac_rotate_step = random.randrange(1, MAC_ROTATE_STEPS_TOTAL + 1)
        self.rpi_rotate_step = random.randrange(RPI_ROTATE_STEPS_MIN, RPI_ROTATE_STEPS_MAX + 1)
        self.advertise_step = random.randrange(ADVERTISE_MIN, ADVERTISE_MAX + 1)

    def _gen_EXPOSURE_NOTIFICATION_TIMER_ANDROID(self):
        """Generate timers according to observed Android behavior"""
        self.mac_rotate_step = random.randrange(1, MAC_ROTATE_STEPS_TOTAL + 1)
        self.rpi_rotate_step = random.randrange(RPI_ROTATE_STEPS_MIN, RPI_ROTATE_STEPS_MAX + 1)
        self.advertise_step = random.randrange(ADVERTISE_GALAXY_S10_MIN, ADVERTISE_GALAXY_S10_MAX + 1)

    def _gen_EXPOSURE_NOTIFICATION_TIMER_ANDROID_R(self):
        """Generate timers according to reported Android R behavior"""
        self.mac_rotate_step = random.randrange(RPI_ROTATE_STEPS_MIN, RPI_ROTATE_STEPS_MAX + 1)
        self.rpi_rotate_step = random.randrange(RPI_ROTATE_STEPS_MIN, RPI_ROTATE_STEPS_MAX + 1)
        self.advertise_step = random.randrange(ADVERTISE_GALAXY_S10_MIN, ADVERTISE_GALAXY_S10_MAX + 1)

    def _gen_EXPOSURE_NOTIFICATION_TIMER_ANDROID_R_FIXED(self):
        """Generate timers according to reported Android R behavior with dual rotation fixed"""
        self.mac_rotate_step = random.randrange(RPI_ROTATE_STEPS_MIN, RPI_ROTATE_STEPS_MAX + 1)
        self.rpi_rotate_step = random.randrange(RPI_ROTATE_STEPS_MIN, RPI_ROTATE_STEPS_MAX + 1)
        self.advertise_step = random.randrange(ADVERTISE_GALAXY_S10_MIN, ADVERTISE_GALAXY_S10_MAX + 1)
        self.is_fixed = True

    def _gen_EXPOSURE_NOTIFICATION_TIMER_SPEC_FIXED(self):
        """Generate timers according to Google/Apple spec with dual rotation fixed"""
        self.mac_rotate_step = random.randrange(1, MAC_ROTATE_STEPS_TOTAL + 1)
        self.rpi_rotate_step = random.randrange(RPI_ROTATE_STEPS_MIN, RPI_ROTATE_STEPS_MAX + 1)
        self.advertise_step = random.randrange(ADVERTISE_MIN, ADVERTISE_MAX + 1)
        self.is_fixed = True

    def _rotate_mac(self):
        """Rotate MAC address"""
        self._mac = uuid.uuid4()
        if self.is_fixed: # fix for dual rotation not taking place currently in this direction
            self._rpi = uuid.uuid4()

    def _rotate_rpi(self):
        """Rotate rolling proximity identifier"""
        self._rpi = uuid.uuid4()
        self._mac = uuid.uuid4()

    def _advertise(self):
        x, y = self._space_map.locate_agent(self._guid)
        self._time_series_data.add((self._step, self._mac, self._rpi, x, y))


class DeviceOwnerAgent(Agent):
    """An agent carrying an ExposureNotification enabled Apple/Google device"""
    def __init__(self, unique_id, model, timers):
        super().__init__(unique_id, model)
        self._timers = timers

    def step(self):
        """Execute agent step"""
        self._timers.step()


class DeviceClusterModel(Model):
    """A model with a single or multiple beacons observing a device cluster of given size"""
    def __init__(self, cluster_size: int, timer_type: ExposureNotificationTimerTypes, time_series_data: TimeSeriesData, space_map: Map):
        self.num_agents = cluster_size
        self.schedule = RandomActivation(self)
        for i in range(self.num_agents):
            space_map.place_agent(i)
            timers = ExposureNotificationTimers(timer_type, guid=i, time_series_data=time_series_data, space_map=space_map)
            a = DeviceOwnerAgent(i, model=self, timers=timers)
            self.schedule.add(a)

    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()


def run_game(n_simulation_runs: int, cluster_size: int, triangulated: bool, timer_type: ExposureNotificationTimerTypes):
    """Run game with parameters, returns trackable fraction of devices per run"""
    assert n_simulation_runs > 0
    assert cluster_size > 0

    global map_movie, action_list, movement_list
    map_movie = list ()
    action_list = deque()
    movement_list = deque()

    trackables = []

    if timer_type in [
            ExposureNotificationTimerTypes.EXPOSURE_NOTIFICATION_TIMER_SPEC,
            ExposureNotificationTimerTypes.EXPOSURE_NOTIFICATION_TIMER_SPEC_FIXED
        ]:
        
        advertising_interval = ADVERTISE_MAX

    elif timer_type in [
            ExposureNotificationTimerTypes.EXPOSURE_NOTIFICATION_TIMER_ANDROID,
            ExposureNotificationTimerTypes.EXPOSURE_NOTIFICATION_TIMER_ANDROID_R,
            ExposureNotificationTimerTypes.EXPOSURE_NOTIFICATION_TIMER_ANDROID_R_FIXED
        ]:

        advertising_interval = ADVERTISE_GALAXY_S10_MAX
    
    else:
        assert False

    for _ in range(0, n_simulation_runs):
        space_map = Map(cluster_size, cluster_size + 1)
        time_series_data = TimeSeriesData()
        model = DeviceClusterModel(cluster_size,
                                timer_type=timer_type,
                                time_series_data=time_series_data,
                                space_map=space_map)

        for _ in range(0, SIMULATION_RUN_STEPS):
            model.step()
            space_map.step()

        trackables += [time_series_data.trackable_fraction(cluster_size=cluster_size,
                                                           advertising_interval=advertising_interval)]

    return trackables

def render_game(n_frames: int, frame_time_steps: int, cluster_size: int):
    """Render game with parameters, returns trackable fraction of devices per run"""

    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    def chunks(s, n):
        for start in range(0, len(s), n):
            yield s[start:start+n]

    def do_render(ax, data, step, prev_step, is_event: bool):
        ax.cla()
        ax.set_title("step {}".format(step))
        ax.imshow(data, cmap='hot', interpolation='nearest')

        EVENT_DURATION = 300
        if is_event:
            pause_duration = EVENT_DURATION
        else:
            pause_duration = max(EVENT_DURATION, (step - prev_step) / 10 - EVENT_DURATION)

        plt.pause(pause_duration)

    aidx = 0
    prev_step = None
    for step, map_str in map_movie:
        cmap = chunks(map_str, math.sqrt(len(map_str)))
        if aidx in action_list:
            if action_list[aidx][0] < step:

                frame_offset = step * frame_time_steps
                if len(action_list) > 0 and frame_offset > action_list[0]:

                    dmap = list.copy(cmap)

                    def traverse_rpi(x, y):
                        dmap[x][y] = '*'
                        pass

                    def traverse_mac(x, y):
                        dmap[x][y] = '#'
                        pass

                    def temp_lost(x, y):
                        dmap[x][y] = '?'
                        pass

                    def recover_temporal(x, y):
                        dmap[x][y] = 'T'
                        pass

                    def recover_spatial(x0, y0, x1, y1):
                        dmap[x0][y0] = 'S'
                        dmap[x1][y1] = 'S'
                        pass

                    def track_lost(x, y):
                        dmap[x][y] = ' '
                        pass

                    {
                        'traverse_rpi': traverse_rpi,
                        'traverse_mac': traverse_mac,
                        'temp_lost': temp_lost,
                        'recover_temporal': recover_temporal,
                        'recover_spatial': recover_spatial,
                        'track_lost': track_lost
                    }[action_list[0][1]](*action_list[0][2:])

                    do_render(ax, dmap, step, prev_step, is_event=True)
                    do_render(ax, cmap, step, prev_step, is_event=False)

                    aidx += 1
                    prev_step = step

    plt.show()

    #
    # non-spatial (globals)
    #
    # d1  d2  d3
    #-d4--d5- d6
    #
    #
    # spatial (geo-buckets)
    # 
    # +--+--+--+--+
    # |d1|d2|d3|d4|
    # +  +  +  +  +
    # |  |  |  |d6|    movement = move/loc switch
    # +--+--+--+--+
