#!/opt/local/bin/python2.7
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import cm
from matplotlib import rc
import utilities

def get_data():
  dir_name = 'energy1e_5/'
  file_name_stub = 'Alpha_VxVyXmodLambda_'
  p615 = pickle.load(open(dir_name + file_name_stub + '615.pdf.pkl', 'r'))
  p820 = pickle.load(open(dir_name + file_name_stub + '820.pdf.pkl', 'r'))
  p1025 = pickle.load(open(dir_name + file_name_stub + '1025.pdf.pkl', 'r'))
  p2047 = pickle.load(open(dir_name + file_name_stub + '2047.pdf.pkl', 'r'))
  return (p615, p820, p1025, p2047)

def remove_tick_string(ax, tick_locs, index, direction):
  if direction == 0:
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_locs)
    labels = ax.get_xticklabels()
  elif direction == 1:
    ax.set_yticks(tick_locs)
    ax.set_yticklabels(tick_locs)
    labels = ax.get_yticklabels()

  assert len(labels) == len(tick_locs)

  indices = np.ones(len(labels)) > 0
  indices[index] = False
  new_labels = list()
  for i, b in enumerate(indices):
    if b:
      new_labels.append(labels[i])

  if direction == 0:
    ax.set_xticks(tick_locs[indices])
    ax.set_xticklabels(new_labels, rotation=30)
  elif direction == 1:
    ax.set_yticks(tick_locs[indices])
    ax.set_yticklabels(new_labels, rotation=30)

def add_omega_over_k(ax):
    ax.text(-0.67, 1.53, r'$\frac{\omega_i}{k_i}$', fontsize=utilities.get_fontsize())

def make_fig(fos):
  utilities.set_rcParams()
  fos = list(fos)
  fig, axs = plt.subplots(2,2)
  for i, fo in enumerate(fos):
    fo.set_columns(1.75)
    fos[i] = fo
  utilities.set_fig_size(fos[0], fig)
  plt.cla()
  size_inches = fig.get_size_inches()
  fig.set_size_inches(size_inches[0], size_inches[0])

  labels = (r'(i)', r'(ii)', r'(iii)', r'(iv)')
  for i, fo in enumerate(fos):
    ax1 = axs[i/2][i%2]
    ax1.tick_params(axis='x', labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)

    tick_locs = np.linspace(-1.5, 1.5, 7)
    if i == 2:
      ax1.set_xticklabels(tick_locs, rotation=30)
      ax1.set_xlabel(fo.xlabel())
    elif i == 3:
      ax1.set_xticklabels(tick_locs, rotation=30)
      remove_tick_string(ax1, tick_locs, 0, 0)
      ax1.set_xlabel(fo.xlabel())
    else:
      ax1.set_xlabel('')
      ax1.set_xticklabels('')

    if i == 0:
      ax1.set_yticklabels(tick_locs, rotation=30)
      remove_tick_string(ax1, tick_locs, 0, 1)
      ax1.set_ylabel(fo.ylabel())
    elif i == 2:
      ax1.set_yticklabels(tick_locs, rotation=30)
      ax1.set_ylabel(fo.ylabel())
    else:
      ax1.set_ylabel('')
      ax1.set_yticklabels('')

    if i == 0 or i == 1:
      add_omega_over_k(ax1)

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)

    pd = fo.plot_data()
    sc = ax1.scatter(pd.xdata(), pd.ydata(), c=pd.zdata(), 
        s=1, alpha=0.25, edgecolors='none', cmap=cm.hsv)
    x0 = -0.578510289369648
    ax1.plot([x0, x0], [-1.5, 1.5], '--', lw=1, color='gray', alpha=0.5)

    ax1.text(-1.45, 1.25, labels[i], fontsize=utilities.get_fontsize())

    if i == 3:
      d = 0.1
      position = fig.add_axes([0.92, d, d/7, 1-2*d])
      legend_ticks = np.linspace(0.0, 1.0, 11)
      locs = [1.0e-3, 0.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0-1.0e-3]
      labs = np.linspace(0.0, 1.0, 11)
      cb = fig.colorbar(sc, cax = position)
      cb.set_alpha(1.0)
      cb.formatter.set_powerlimits((0, 0))
      cb.set_ticks(locs)
      cb.set_ticklabels(labs)
      #cb.update_ticks()
      cb.ax.tick_params(labelsize=6) 
      cb.draw_all()

  fig.subplots_adjust(hspace=0)
  fig.subplots_adjust(wspace=0)

  fig.savefig('tiled_vx_vy_xmodlambda.pdf', bbox_inches='tight', dpi=500)
  plt.close()

def main():
  fos = get_data()
  make_fig(fos)

if __name__ == "__main__":
  main()






















