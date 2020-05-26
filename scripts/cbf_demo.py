import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

df = pd.read_csv('log_2020_03_10_02_16_51_trial_1.csv')
df[['x_pitch', 'x_roll']] = np.rad2deg(df[['x_pitch', 'x_roll']])
d = df[['time', 'x_pitch', 'x_roll']].query('time <= 60')
d.columns = ['Time', 'Pitch', 'Roll']
d['Feedback'] = d.Pitch.abs() > 3
d.Feedback = d.Feedback.astype(int)

def cbf_demo():
    # f, ax = plt.subplots(2, 1, figsize=(16, 9), dpi=300, sharex=True)

    # ax[0].tick_params(
    #     axis='both',
    #     which='both',
    #     bottom=False,
    #     top=False,
    #     left=False,
    #     right=False,
    #     labelbottom=False,
    #     labeltop=False,
    #     labelleft=False,
    #     labelright=False,
    # )

    # ax[1].tick_params(
    #     axis='both',
    #     which='both',
    #     bottom=False,
    #     top=False,
    #     left=False,
    #     right=False,
    #     labelbottom=False,
    #     labeltop=False,
    #     labelleft=False,
    #     labelright=False,
    # )

    # d.plot(x='Time', y='Pitch', ax=ax[0], legend=None, color='k')
    # d.plot.scatter(x='Time', y='Feedback', ax=ax[1], legend=None, color='k', s=1)

    # ax[0].set_ylabel('Pitch', size=18)
    # ax[1].set_ylabel('Feedback', size=18)

    # ax[0].axhspan(-12, -3, alpha=0.125, facecolor='k')
    # ax[0].axhspan(3, 12, alpha=0.125, facecolor='k')

    # ax[0].set_xlim((0, 60))
    # ax[0].set_ylim((-12, 12))
    # ax[1].set_ylim((0.8, 1.2))
    # # plt.show()

    f, ax = plt.subplots(1, 1, figsize=(16/3, 9/3), dpi=300)

    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )

    ax.plot(d.Time, d.Pitch, 'k')

    ax.set_ylabel('Signal', size=18)
    ax.set_xlabel('Time', size=18)

    ax.set_xlim((0, 60))
    ax.set_ylim((-11, 11))
    plt.tight_layout()
    # plt.show()
    # plt.savefig('signal.pdf')
    # plt.savefig('signal.png')
    # plt.close('all')

    f, ax = plt.subplots(1, 1, figsize=(16/3, 9/3), dpi=300)

    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )

    ax.plot(d.Time, d.Pitch, 'k')

    ax.set_ylabel('Signal', size=18)
    ax.set_xlabel('Time', size=18)

    ax.axhspan(-3, -12, alpha=0.125, facecolor='k')
    ax.axhspan(3, 12, alpha=0.125, facecolor='k')

    ax.axhline(-3, alpha=1, color='k')
    ax.axhline(3, alpha=1, color='k')

    ax.set_xlim((0, 60))
    ax.set_ylim((-11, 11))
    plt.tight_layout()
    # plt.show()
    # plt.savefig('signal_w_bandwidth.pdf')
    # plt.savefig('signal_w_bandwidth.png')
    # plt.close('all')

    f, ax = plt.subplots(1, 1, figsize=(16/3, 9/3), dpi=300)

    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )

    supper = np.ma.masked_where(d.Pitch <  3, d.Pitch)
    slower = np.ma.masked_where(d.Pitch > -3, d.Pitch)
    smiddle = np.ma.masked_where((d.Pitch < -3) | (d.Pitch > 3), d.Pitch)
    ax.plot(d.Time, smiddle, 'g', d.Time, slower, 'r', d.Time, supper, 'r')

    ax.set_ylabel('Signal', size=18)
    ax.set_xlabel('Time', size=18)

    ax.axhspan(-3, -12, alpha=0.125, facecolor='k')
    ax.axhspan(3, 12, alpha=0.125, facecolor='k')

    ax.axhline(-3, alpha=1, color='k')
    ax.axhline(3, alpha=1, color='k')

    ax.set_xlim((0, 60))
    ax.set_ylim((-11, 11))
    plt.tight_layout()
    # plt.show()
    # plt.savefig('signal_w_feedback.pdf')
    # plt.savefig('signal_w_feedback.png')
    # plt.close('all')

###############################################################################


def note_percent(angle, i=None):
    y = np.interp(angle, cdf.Pitch.values, cdf.CDF.values)

    if i != None:
        ax[i].axvline(angle, ymin=0, ymax=y, color='k', alpha=0.5, ls='--')
        ax[i].annotate("{:.2f}%".format(y)[2:], (angle, y + 0.01))
    else:
        ax.axvline(angle, ymin=0, ymax=y, color='k', alpha=0.5, ls='--')
        ax.annotate("{:.2f}%".format(y)[2:], (angle, y + 0.01))



f, ax = plt.subplots(1, 3, figsize=(16,4), dpi=100, sharex=True, sharey=True)
res = ax[0].hist(df.pitch.abs().values, bins=np.linspace(0, 10, 1000), density=True, histtype='step', cumulative=-1, color='b')
# plt.xlim(0, 10)
# plt.ylim(0, 1)
ax[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
# plt.xlabel('Feedback Bandwidth', size=18)
# plt.ylabel('Percent', size=18)

cdf = pd.DataFrame(res).T[[0, 1]].dropna()
cdf.columns = ['CDF', 'Pitch']
cdf = cdf.astype(float)

note_percent(2, 0)
note_percent(3, 0)
note_percent(4, 0)

ax[0].set_title('No Input')
# plt.tight_layout()
# plt.show()
# plt.savefig('percent_on.pdf')
# plt.savefig('percent_on.png')
# plt.close('all')

###############################################################################

df = pd.read_csv('trained.csv', header=None)
df.columns = ['time', 'pitch']

# f, ax = plt.subplots(figsize=(16/3,16/3), dpi=300)
res = ax[2].hist(df.pitch.abs().values, bins=np.linspace(0, 10, 1000), density=True, histtype='step', cumulative=-1, color='b')
# plt.xlim(0, 10)
# plt.ylim(0, 1)
# ax[2].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
# plt.xlabel('Feedback Bandwidth', size=18)
# plt.ylabel('Percent', size=18)

cdf = pd.DataFrame(res).T[[0, 1]].dropna()
cdf.columns = ['CDF', 'Pitch']
cdf = cdf.astype(float)

note_percent(2, 2)
note_percent(3, 2)
note_percent(4, 2)

ax[2].set_title('Model')
# plt.tight_layout()
# plt.show()
# plt.savefig('percent_on.pdf')
# plt.savefig('percent_on.png')
# plt.close('all')

###############################################################################

df = pd.read_csv('feedback_P_34.csv')

# f, ax = plt.subplots(figsize=(16/3,16/3), dpi=300)

r = []
for subject in df.subject.unique():
    res = ax[1].hist(df.query('subject == @subject').pitch.abs().values, bins=np.linspace(0, 10, 1000), density=True, histtype='step', cumulative=-1, alpha=0.05, color='k')
    cdf = pd.DataFrame(res).T[[0, 1]].dropna()
    cdf.columns = ['CDF', 'Pitch']
    cdf = cdf.astype(float)
    cdf['subject'] = subject
    r.append(cdf)

cdf = pd.concat(r)
cdf = cdf.groupby('Pitch').CDF.agg(['mean', 'sem']).reset_index()
cdf.columns = ['Pitch', 'CDF', 'sem']

ax[1].plot(cdf.Pitch, cdf['CDF'], alpha=1.0, color='b')
ax[1].fill_between(cdf.Pitch, cdf['CDF'] - cdf['sem'], cdf['CDF'] + cdf['sem'], color='b', alpha=0.25)

plt.xlim(0, 10)
plt.ylim(0, 1)
# ax[1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax[1].set_xlabel('Feedback Bandwidth', size=18)
ax[0].set_ylabel('Percent', size=18)

note_percent(2, 1)
note_percent(3, 1)
note_percent(4, 1)

# plt.suptitle("Percent of Time Feedback Active")
ax[1].set_title('Experiment')
plt.tight_layout()
plt.show()
# plt.savefig('percent_on.pdf')
# plt.savefig('percent_on.png')
# plt.close('all')

###############################################################################

df = pd.read_csv('log_2020_03_10_02_16_51_trial_1.csv')

f, ax = plt.subplots(figsize=(16/3,9/3), dpi=300)
res = ax.hist(df.pitch.abs().values, bins=np.linspace(0, 10, 1000), density=True, histtype='step', cumulative=-1, color='b')
plt.xlim(0, 10)
plt.ylim(0, 1)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
plt.xlabel('Feedback Bandwidth')
plt.ylabel('Percent')

cdf = pd.DataFrame(res).T[[0, 1]].dropna()
cdf.columns = ['CDF', 'Pitch']
cdf = cdf.astype(float)

note_percent(2)
note_percent(3)
note_percent(4)

plt.tight_layout()
# plt.show()
plt.savefig('no_input_feedback_on.pdf')
plt.savefig('no_input_feedback_on.png')
plt.close('all')

df = pd.read_csv('feedback_P_34.csv')

f, ax = plt.subplots(figsize=(16/3,9/3), dpi=300)

r = []
for subject in df.subject.unique():
    res = ax.hist(df.query('subject == @subject').pitch.abs().values, bins=np.linspace(0, 10, 1000), density=True, histtype='step', cumulative=-1, alpha=0.05, color='k')
    cdf = pd.DataFrame(res).T[[0, 1]].dropna()
    cdf.columns = ['CDF', 'Pitch']
    cdf = cdf.astype(float)
    cdf['subject'] = subject
    r.append(cdf)

cdf = pd.concat(r)
cdf = cdf.groupby('Pitch').CDF.agg(['mean', 'sem']).reset_index()
cdf.columns = ['Pitch', 'CDF', 'sem']

ax.plot(cdf.Pitch, cdf['CDF'], alpha=1.0, color='b')
ax.fill_between(cdf.Pitch, cdf['CDF'] - cdf['sem'], cdf['CDF'] + cdf['sem'], color='b', alpha=0.25)

plt.xlim(0, 10)
plt.ylim(0, 1)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
ax.set_xlabel('Feedback Bandwidth')
ax.set_ylabel('Percent')

note_percent(2)
note_percent(3)
note_percent(4)

plt.tight_layout()
# plt.show()
plt.savefig('p34_feedback_on.pdf')
plt.savefig('p34_feedback_on.png')
plt.close('all')
