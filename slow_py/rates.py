import matplotlib.pyplot as plt
import datetime
from catalogs import *
from csep.utils import plots
import csep
from pycsep.csep.utils.calc import bin1d_vec
from pycsep.csep.utils.time_utils import datetime_to_utc_epoch

regions = [
    'japan',
    # 'mexico',
    # 'chile',
    # 'alaska',
    # 'cascadia'
]
catalogs = {
    'japan': get_japan_catalog(depth=150),
    'mexico': get_mexico_catalog(depth=150),
    'chile': get_chile_catalog(depth=150),
    'alaska': get_alaska_catalog(depth=150),
    'cascadia': get_cascadia_catalog(depth=150)
}
time_extent = {
    'japan': [None, None],
    'mexico': [datetime.datetime(1997, 6, 1), datetime.datetime(2020, 1, 1)],
    'chile': [datetime.datetime(2006, 1, 1), datetime.datetime(2016, 1, 1)],
    'alaska': [datetime.datetime(2007, 1, 1), datetime.datetime(2022, 1, 1)],
    'cascadia': [datetime.datetime(2004, 1, 1), datetime.datetime(2018, 1, 1)],
}
min_mag = 5
eq_size = 1


def _plot_cumulative_events_versus_time(
        observation: "CSEPCatalog",
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        time_axis: str = "datetime",
        bins: int = 200,

        sim_label= "Simulated",
        obs_label= "Observation",
        ax = None,
        show: bool = False,
        **kwargs,
):
    # Initialize plot

    plot_args = {**plots.DEFAULT_PLOT_ARGS, **kwargs.get("plot_args", {}), **kwargs}
    fig, ax = plt.subplots(figsize=plot_args["figsize"]) if ax is None else (ax.figure, ax)

    # offsets to start at 0 time and converts from millis to hours
    if start_time is None:
        start_time = np.min(observation.get_datetimes())
    if end_time is None:
        end_time = np.min(observation.get_datetimes())
    start_time = datetime_to_utc_epoch(start_time)
    end_time = datetime_to_utc_epoch(end_time)

    time_bins, dt = np.linspace(start_time, end_time, bins, endpoint=True, retstep=True)
    n_bins = time_bins.shape[0]

    # compute median for comcat data
    obs_binned_counts = np.zeros(n_bins)

    inds = bin1d_vec(observation.get_epoch_times(), time_bins)
    for j in range(observation.event_count):
        obs_binned_counts[inds[j]] += 1
    obs_summed_counts = np.cumsum(obs_binned_counts)

    # make all arrays start at zero
    time_bins = np.insert(time_bins, 0, 2 * time_bins[0] - time_bins[1])  # One DT before

    if time_axis == "datetime":
        time_bins = [csep.epoch_time_to_utc_datetime(i) for i in time_bins]
        ax.xaxis.set_major_locator(plot_args["datetime_locator"])
        ax.xaxis.set_major_formatter(plot_args["datetime_formatter"])
        ax.set_xlabel(plot_args["xlabel"] or "Datetime", fontsize=plot_args["xlabel_fontsize"])
        fig.autofmt_xdate()

    # Plotting
    ax.plot(
        time_bins[:-1],
        obs_summed_counts,
        color="black",
        linewidth=plot_args["linewidth"],
        label=obs_label,
    )




    return ax




def plot_sse_box(sse, ax=None):

    sse_starttimes = np.array([datetime.datetime.fromisoformat(i) for i in sse.time])
    duration = sse.duration

    duration[np.isnan(duration)] = np.nanmin(duration)

    sse_endtimes = np.array([i + datetime.timedelta(seconds=j) for i, j in zip(sse_starttimes, duration)])

    ax.barh(
        y=min_mag,
        width=[i for i in (sse_endtimes - sse_starttimes)],  # Convert to days
        left=sse_starttimes,
        height=sse.mag - min_mag,  # Height of the bar
        color='green',  # Random color
        alpha=0.2,  # Transparency
        align='edge',
        label='SSE'
    )
    return ax
    # ax.plot(sse_datetimes, sse.mag, '.')

# ### Rates
# for region in regions:
#     sse = get_slow_slip_catalog_region(region)
#     cat = catalogs[region]
#     ax = plots.plot_magnitude_versus_time(cat.filter(f'magnitude >= {min_mag}'), figsize=(12, 4),
#                                           size=eq_size)
#
#     ax = plot_sse_box(sse, ax)
#
#     ax.set_ylim([min_mag, None])
#     ax.set_xlim(time_extent[region][0], time_extent[region][1])
#     ax.set_title(region.capitalize())
#     plt.tight_layout()
#     plt.savefig(f'../images/rates/{region}_rates_{min_mag}.png')


## Cumulative rates
for region in regions:
    sse = get_slow_slip_catalog_region(region)
    cat = catalogs[region]
    ax = _plot_cumulative_events_versus_time(cat, time_extent[region][0], time_extent[region][1])
    #
    # ax = plot_sse_box(sse, ax)
    #
    # ax.set_ylim([min_mag, None])
    # ax.set_xlim(time_extent[region][0], time_extent[region][1])
    # ax.set_title(region.capitalize())
    # plt.tight_layout()
    plt.savefig(f'../images/rates/{region}_cumrates_{min_mag}.png')
