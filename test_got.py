import uuid
import math
from gameoftrackers import TimeSeriesData, ExposureNotificationTimers, ExposureNotificationTimerTypes, SIMULATION_RUN_STEPS, DeviceOwnerAgent, run_game, Map

def test_got_TimeSeriesData():

    sx0 = 0
    macx0 = uuid.uuid4()
    rpix0 = uuid.uuid4()

    sy0 = 2
    macy0 = uuid.uuid4()
    rpiy0 = uuid.uuid4()

    sz0 = 6    
    macz0 = uuid.uuid4()
    rpiz0 = uuid.uuid4()

    sx1 = 10
    macx1 = macx0
    rpix1 = uuid.uuid4()
    
    sy1 = 12
    macy1 = uuid.uuid4()
    rpiy1 = rpiy0
    
    sz1 = 16
    macz1 = uuid.uuid4() # full rotation at 16
    rpiz1 = uuid.uuid4()

    sx2 = 20
    macx2 = uuid.uuid4() # full rotation at 20
    rpix2 = uuid.uuid4()
    
    sy2 = 22
    macy2 = macy1
    rpiy2 = uuid.uuid4()
    
    sz2 = 26
    macz2 = uuid.uuid4()
    rpiz2 = rpiz1

    t = TimeSeriesData()
    t.add((sx0, macx0, rpix0, -1, -1))
    t.add((sy0, macy0, rpiy0, -1, -1))
    t.add((sz0, macz0, rpiz0, -1, -1))
    t.add((sx1, macx1, rpix1, -1, -1))
    t.add((sy1, macy1, rpiy1, -1, -1))
    t.add((sz1, macz1, rpiz1, -1, -1))
    t.add((sx2, macx2, rpix2, -1, -1))
    t.add((sy2, macy2, rpiy2, -1, -1))
    t.add((sz2, macz2, rpiz2, -1, -1))

    fraction = t.trackable_fraction(3, 10)

    assert fraction >= .66665 and fraction <= .66667

def test_got_ExposureNotificationTimers():

    space_map = Map(1, 2)
    space_map.place_agent(0)
    time_series_data = TimeSeriesData()
    timers = ExposureNotificationTimers(ExposureNotificationTimerTypes.EXPOSURE_NOTIFICATION_TIMER_SPEC, guid=0, time_series_data=time_series_data, space_map=space_map)

    for _ in range(SIMULATION_RUN_STEPS):
        timers.step()

    MIN_ADVERTISINGS = math.floor(SIMULATION_RUN_STEPS / 3)
    MAX_ADVERTISINGS = math.floor(SIMULATION_RUN_STEPS / 2)
    l = len(time_series_data.data)
    assert l >= MIN_ADVERTISINGS and l <= MAX_ADVERTISINGS

def test_got_run_game():
    for triangulated in [False, True]:
        trackables = run_game(n_simulation_runs=1,
                            cluster_size=3,
                            triangulated=triangulated,
                            timer_type=ExposureNotificationTimerTypes.EXPOSURE_NOTIFICATION_TIMER_SPEC)
        assert len(trackables) >= 0 and len(trackables) < 3
        assert trackables[0] >= .99999

def test_got_render_game():
    pass

def test_got_enriched_graph():
    pass

def test_elasticsearch():
    pass

def test_deep_got():
    # supervised training on identified cases
    # monte carlo? score adjustments?
    # just spec
    pass