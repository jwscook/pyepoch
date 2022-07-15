#!/opt/local/bin/python2.7
import pickle
import numpy as np
import scipy as sp
import random
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import cm
import os
import sys
import re
from scipy import signal
from numpy import *
from sdf import *

epsilon0 = 8.854187817e-12
mu0 = pi*4e-7
q0 = 1.60217657e-19
me = 9.10938291e-31
c0 = 1.0/np.sqrt(epsilon0*mu0)
ion_string_1 = 'Deuteron'
ion_string_latex_1 = 'D'
ion_string_2 = 'Alpha'
ion_string_latex_2 = '\alpha'
#ion_string_2 = 'Proton'
#ion_string_latex_2 = 'P'

def get_fontsize():
  return 8 
  
def set_rcParams():
  plt.rcParams['text.latex.unicode']=True
  plt.rcParams['text.usetex']=True
  plt.rcParams['pgf.texsystem'] = 'pdflatex'
  plt.rcParams['font.family'] = 'serif'
  plt.rcParams['axes.labelsize'] = get_fontsize()
  plt.rcParams['axes.titlesize'] = get_fontsize()
  plt.rcParams['legend.fontsize'] = get_fontsize()
  plt.rcParams['xtick.labelsize'] = get_fontsize()
  plt.rcParams['ytick.labelsize'] = get_fontsize()

def golden_ratio():
  return (sqrt(5)-1.0)/2.0 # Aesthetic ratio

class plot_options:
  _data_name = "empty"
  _sdfkey = "empty"
  _plot_types = {"empty"}
  _data = 0
  def __init__(self, name, sdfkey, plot_types, data):
    self._name = name
    self._sdfkey = sdfkey
    self._plot_types = plot_types
    self._data = data
  def data(self):
    return self._data
  def name(self):
    return self._name
  def sdfkey(self):
    return self._sdfkey
  def plot_types(self):
    return self._plot_types
  def set_data(self, data):
    self._data = data
  def set_name(self, name):
    self._name = name
  def set_sdfkey(self, sdfkey):
    self._sdfkey = sdfkey
  def set_plot_types(self, plot_types):
    self._plot_types = plot_types

class figure_options:
  _title = "title"
  _xlabel = "xlabel"
  _ylabel = "ylabel"
  _filename = ""
  _columns = 2
  _plot_data = None
  _legend_type = 0
  _projection = None
  _method = "imshow"
  def __init__(self, title, xlabel, ylabel, filename, \
      save_display=0, columns=2, legend_type = 0, projection=None, \
      method="imshow"):
    self._title = title
    self._xlabel = xlabel
    self._ylabel = ylabel
    self._filename = filename
    self._columns = columns
    self._save_display = save_display
    self._legend_type = legend_type
    self._projection = projection
    self._method = method
  def title(self):
    return self._title
  def xlabel(self):
    return self._xlabel
  def ylabel(self):
    return self._ylabel
  def filename(self):
    return self._filename
  def columns(self):
    return self._columns
  def save_display(self):
    return self._save_display
  def plot_data(self):
    return self._plot_data
  def legend_type(self):
    return self._legend_type
  def projection(self):
    return self._projection
  def method(self):
    return self._method
  def set_title(self, title):
    self._title = title
  def set_xlabel(self, xlabel):
    self._xlabel = xlabel
  def set_ylabel(self, ylabel):
    self._ylabel = ylabel
  def set_filename(self, filename):
    self._filename = filename
  def set_columns(self, columns):
    self._columns = columns
  def set_save_display(self, save_display):
    self._save_display = save_display
  def set_plot_data(self, plot_data):
    self._plot_data = plot_data
  def set_legend_type(self, legend_type):
    self._legend_type = legend_type
  def set_projection(self, projection):
    self._projection = projection
  def set_method(self, method):
    self._method = method

class plot_data:
  _dim = 1
  _xdata = ()
  _ydata = ()
  _zdata = ()
  _ldata = ()
  _ndata = 0
  def __init__(self, dim=1, xdata=None, ydata=None, zdata=None, ldata=None):
    self._dim = dim
    if xdata is not None:
      self._xdata = (xdata,)
    if ydata is not None:
      self._ydata = (ydata,)
    if zdata is not None:
      self._zdata = (zdata,)
    if ldata is not None:
      self._ldata = (ldata,)
    if xdata is not None:
      self._ndata = 1
  def dim(self):
    return self._dim
  def xdata(self):
    return self._xdata
  def ydata(self):
    return self._ydata
  def zdata(self):
    return self._zdata
  def ldata(self):
    return self._ldata
  def data(self, index=None):
    if index is None:
      if self._dim == 1:
        return self._xdata, self._ydata, self._ldata
      if self._dim == 2:
        return self._xdata, self._ydata, self._zdata, self._ldata
    else:
      assert(index < self._ndata)
      i = int(index)
      if self._dim == 1:
        return self._xdata[i], self._ydata[i], self._ldata[i]
      if self._dim == 2:
        return self._xdata[i], self._ydata[i], self._zdata[i], self._ldata[i]
  def xyzdata(self):
    return (self._xdata, self._ydata, self._zdata, self._ldata)
  def ldata(self):
    return self._ldata
  def ndata(self):
    return self._ndata
  def add_xydata(self, xdata, ydata, ldata=""):
    assert(self._dim == 1)
    assert(len(xdata) == len(ydata))
    if self._ndata == 0:
      self._xdata = (xdata,)
      self._ydata = (ydata,)
      self._ldata = (ldata,)
      self._ndata = 1
    else:
      self._xdata = self._xdata + (xdata,)
      self._ydata = self._ydata + (ydata,)
      self._ldata = self._ldata + (ldata,)
      self._ndata += 1
  def add_xyzdata(self, xdata, ydata, zdata, ldata=""):
    assert(self._dim >= 2)
    if self._ndata == 0:
      self._xdata = (xdata,)
      self._ydata = (ydata,)
      self._zdata = (zdata,)
      self._ldata = (ldata,)
      self._ndata = 1
    else:
      self._xdata = self._xdata + (xdata,)
      self._ydata = self._ydata + (ydata,)
      self._zdata = self._zdata + (zdata,)
      self._ldata = self._ldata + (ldata,)
      self._ndata += 1
  def __add__(self, other):
    assert(self._dim == other._dim)
    out = plot_data(self._dim)
    assert out._ndata == 0
    for i in range(self._ndata):
      if self._dim == 1:
        (x, y, l) = self.data(i)
        out.add_xydata(x, y, l)
      if self._dim == 2:
        (x, y, z, l) = self.data(i)
        out.add_xyzdata(x, y, z, l)
    for i in range(other._ndata):
      if other._dim == 1:
        (x, y, l) = other.data(i)
        out.add_xydata(x, y, l)
      if other._dim == 2:
        (x, y, z, l) = other.data(i)
        out.add_xyzdata(x, y, z, l)
    return out

def sdf_counter(path):
  list_dir = []
  list_dir = os.listdir(path)
  count = 0
  for file in list_dir:
    if file.endswith('.sdf'): 
      count += 1
  #files_in_time_order = os.popen('ls -ht ' + path
  #+ '*.sdf | head -n 1').readlines()
  #last_file_written = "('%s')" % "','".join(files_in_time_order)
  #numbers = re.findall(r'\d+', last_file_written)
  #count = int(numbers[len(numbers)-1])    
  return count

def set_fig_size(fo, fig):
  fig_width = 6.0/fo.columns()
  fig_height = fig_width*golden_ratio()
  fig.set_size_inches(fig_width, fig_height)
  plt.rcParams['figure.figsize'] = [fig_width,fig_height]

def prep_fig(fo):
  set_rcParams()
  d = 0.075
  fig = plt.figure()
  ax1 = fig.add_axes([d, d, 1-2*d, 1-2*d], \
      label="ax1", \
      projection=fo.projection())
  set_fig_size(fo, fig)
  if (fo.save_display() == 2):
    try:
      pd = pickle.load(open('pd.pkl', 'r'))
      fopd = fo.plot_data()
      pd = pd + fopd
      fo.set_plot_data(pd)
    except (IOError):
      pass

  return (fig, ax1, d)

def post_fig(fo, fig, ax1):
  if (fo.save_display() == 0):
    ax1.set_title(fo.title())
    plt.show()
  elif (fo.save_display() == 1):
    fig.savefig(fo.filename(), bbox_inches='tight', dpi=500)
  elif (fo.save_display() == 2):
    fig.savefig(fo.filename(), bbox_inches='tight', dpi=500)
    pickle.dump(fo.plot_data(), open('pd.pkl', 'w'))
  else:
    fig.savefig(fo.filename(), bbox_inches='tight', dpi=500)
    ax1.set_title(fo.title())
    plt.show()

  if fo.save_display() > 0:
    pickle.dump(fo, open(fo.filename() + '.pkl', 'w'))

  plt.close()

def running_mean(y, n_mean):
  assert(n_mean % 2 == 1)
  n = y.size
  ycopy = np.copy(y)
  for i in range(n):
    a = max(0, i-(n_mean-1)/2)
    z = min(n, i+(n_mean-1)/2+1)
    ycopy[i] = np.mean(y[a:z])
  return ycopy

def get_markers():
  return ['o', 'v', 'x', 'v', 's', 'd', '<', '>', '^', 'h', '*'] 
def get_marker(i):
  return get_markers()[i]

def draw_alpha_deuterons(ax):
  ax.plot([6.0, 7.2], [-0.03, -0.03], lw=1.0, c='k', dashes=[10, 2])
  ax.text(7.5, -0.04, r'$\alpha$ particles', fontsize = get_fontsize())
  ax.plot([6.0, 7.2], [0.02, 0.02], lw=1.0, c='k', ls='-')
  ax.text(7.5, 0.01, r'deuterons', fontsize = get_fontsize())

def plot_1d(fo, po):
  if fo.save_display() == -1:
    return
  (fig, ax1, d) = prep_fig(fo)
  plt.cla()
  pd = fo.plot_data()

  number_of_legends = 0
  colours = ['r', 'g', 'g', 'c', 'k', 'm', 'y']
  for i in range(pd.ndata()):
    (x,y,l) = pd.data(i)
    c = plt.cm.jet(float(i)/float(pd.ndata()))
    mi = int(np.floor(i/2.0))
    m = get_marker(mi)
    markers_on=64
    # special plot for stimulated emission prl
    # uncomment from here
    #nd = float(pd.ndata())
    #nc = np.floor(i/2.0) / (nd-2.0) * 2.0
    #c = plt.cm.jet(nc)
    #mi = int(np.floor(i/2.0))
    #m = get_marker(mi)
    #markers_on = [np.where(np.abs(y) > 0.05)[0][0]]
    # to here
    if (l.rfind("$") > -1):
      pass
    else:
      l = re.sub(r'_', ' ', l)
      l = re.sub(r'\W+', '', l)
    if l == "":
      ax1.plot(x, y, color=c, 
        markevery=markers_on, marker=m, markersize=4, mec=c, mfc=c)
    else:
      number_of_legends += 1
      ax1.plot(x, y, label=l, color=c, linestyle="-", dashes=[10, 2], 
        markevery=markers_on, marker=m, markersize=4, mec=c, mfc=c)

  add_legend = number_of_legends != 0

  #draw_alpha_deuterons(ax1)

  ax1.set_xlabel(fo.xlabel())
  ax1.set_ylabel(fo.ylabel())
  #plt.xticks(np.arange(0, 11))
  plt.xticks(np.arange(0, 6))

  if po.xlimits[0] < po.xlimits[1]:
    ax1.set_xlim(po.xlimits[0], po.xlimits[1])
  if po.ylimits[0] < po.ylimits[1]:
    ax1.set_ylim(po.ylimits[0], po.ylimits[1])
  if add_legend and (fo.legend_type() == 1):
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=3, mode="expand", borderaxespad=0., numpoints=1)

#  for (p, s, d) in zip([3.0, 4.0, 5.0, 10.0], [r'(i)', r'(ii)', r'(iii)', r'(iv)'], [0.5, 0.65, 0.8, 0.7]):
#    ax1.plot([p, p], po.ylimits, 'k--', lw=1.0, color='gray', alpha=0.5)
#    ax1.text(p-d, po.ylimits[0]+0.01, s, fontsize = get_fontsize())

  post_fig(fo, fig, ax1)

def plot_2d(x, y, z, fo, po):
  fo.set_plot_data(plot_data(3, x, y, z))

  if fo.save_display() == -1:
    return

  (fig, ax1, d) = prep_fig(fo)
  plt.cla()
  dx = x[1] - x[0]
  dy = y[1] - y[0]

  if fo.method() == "imshow":
    if po.xlimits[0] < po.xlimits[1]:
      z[x < po.xlimits[0], :] *= 0 
      z[x > po.xlimits[1], :] *= 0 
    if po.ylimits[0] < po.ylimits[1]:
      z[:, y < po.ylimits[0]] *= 0 
      z[:, y > po.ylimits[1]] *= 0 
    extent=(x.min()-dx/2.0, x.max()+dx/2.0, y.min()-dy/2.0, y.max()+dy/2.0)
    sc = ax1.imshow(transpose(z), interpolation='nearest', aspect='auto', 
      extent=extent,
      origin='lower')#, rasterize=True) 
    position = fig.add_axes([1-0.75*d, d, d/4, 1-2*d]) 

  elif fo.method() == "scatter":
    sc = ax1.scatter(x, y, c=z, s=1, alpha=0.25, 
        edgecolors='none', cmap=cm.hsv)
    ax1.set_aspect('equal')
    position = fig.add_axes([0.8, d, d/4, 1-2*d]) 
    x0 = -0.578510289369648
    ax1.plot([x0, x0], [-1.5, 1.5], 'k--')
    xlabels = np.linspace(-1.5, 1.5, 7)
    ax1.set_xticklabels(xlabels, rotation=-30)

  cb = fig.colorbar(sc, cax = position)
  cb.set_alpha(1.0)
  cb.formatter.set_powerlimits((0, 0))
  cb.update_ticks()
  cb.draw_all()

  if np.ceil(np.max(np.max(z))) == 1 and np.floor(np.min(np.min(z))) == 0:
    cb.set_clim(0, 1.0)

  ax1.set_xlabel(fo.xlabel())
  ax1.set_ylabel(fo.ylabel())

  ax1.axis('tight')
  if po.xlimits[0] < po.xlimits[1]:
    ax1.set_xlim(po.xlimits[0], po.xlimits[1])
  else:
    ax1.set_xlim(np.min(x), np.max(x))
  if po.ylimits[0] < po.ylimits[1]:
    ax1.set_ylim(po.ylimits[0], po.ylimits[1])
  else:
    ax1.set_ylim(np.min(y), np.max(y))

  # for stimulated emission figs. from here
  lims = ax1.get_xlim()
  #for (p, s) in zip([3.0, 4.0, 5.0, 9.99], [r'(i)', r'(ii)', r'(iii)', r'(iv)']):
  #  ax1.plot([x[0], x[-1]], [p, p], 'k--', lw=1.0, color='gray', alpha=0.5)
  #  ax1.text(lims[0]+0.05, p-0.62, s, fontsize = get_fontsize())
  # tk

  #lims = ax1.get_xlim()
  #for (p, s) in zip([3.0, 4.0, 5.0, 9.99], [r'(i)', r'(ii)', r'(iii)', r'(iv)']):
  #  ax1.plot(lims, [p, p], 'k--', lw=1.0, color='gray', alpha=0.5)
  #  ax1.text(lims[0]+0.01*(lims[1]-lims[0]), p-0.62, s, fontsize = get_fontsize())

  ax1.text(0.94*ax1.get_xlim()[1], 0.93*ax1.get_ylim()[1], r"(b)", fontsize=get_fontsize())
  # to here

  post_fig(fo, fig, ax1)

def plot_polar(r, t, z, fo, po):

  if fo.save_display() == -1:
    return

  (fig, ax1, d) = prep_fig(fo)
  plt.cla()

  T, R = np.meshgrid(t, r)

  sc = ax1.pcolormesh(T, R, z)
  ax1.set_yticks([])

  ax1.set_axis_bgcolor(cm.jet(0))
  #theta_ticks = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
  #ax1.set_thetagrids(theta_ticks, frac=1.175)
  #plt.xticks([np.pi*x for x in theta_ticks], \
  #  [r'$-\pi$', \
  #  r'$-\frac{3\pi}{4}$',\
  #  r'$-\frac{\pi}{2}$',\
  #  r'$-\frac{\pi}{4}$',\
  #  '$0$',\
  #  r'$+\frac{\pi}{4}$',\
  #  r'$+\frac{\pi}{2}$',\
  #  r'$+\frac{3\pi}{4}$'])
  plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')
  ax1.text(1.05, 0.5, r'$0$', \
    ha='center', va='center', transform=ax1.transAxes, fontsize=get_fontsize())
  ax1.text(0.5, 1.05, r'$\frac{\pi}{2}$', \
    ha='center', va='center', transform=ax1.transAxes, fontsize=get_fontsize())
  ax1.text(-0.08, 0.5, r'$\pm\pi$', \
    ha='center', va='center', transform=ax1.transAxes, fontsize=get_fontsize())
  ax1.text(0.48, -0.08, r'$-\frac{\pi}{2}$', \
    ha='center', va='center', transform=ax1.transAxes, fontsize=get_fontsize())
  ax1.text(0.86, 0.05, r'$-\frac{\pi}{4}$', \
    ha='center', va='center', transform=ax1.transAxes, fontsize=get_fontsize())
  ax1.text(0.89, 0.89, r'$\frac{\pi}{4}$', \
    ha='center', va='center', transform=ax1.transAxes, fontsize=get_fontsize())
  ax1.text(0.1, 0.89, r'$\frac{3\pi}{4}$', \
    ha='center', va='center', transform=ax1.transAxes, fontsize=get_fontsize())
  ax1.text(0.1, 0.05, r'$-\frac{3\pi}{4}$', \
    ha='center', va='center', transform=ax1.transAxes, fontsize=get_fontsize())

  position = fig.add_axes([0.825, d, d/4, 1-2*d]) 
  plt.colorbar(sc, cax = position) ## 

  if po.xlimits[0] < po.xlimits[1]:
    ax1.set_xlim(po.xlimits[0], po.xlimits[1])
  if po.ylimits[0] < po.ylimits[1]:
    ax1.set_ylim(po.ylimits[0], po.ylimits[1])

  post_fig(fo, fig, ax1)

def filter_window(matrix_in):
  filter_window = signal.hanning(matrix_in.shape[1])
  filter_window = np.reshape(filter_window, (1, matrix_in.shape[1]))
  filter_window = np.tile(filter_window, (matrix_in.shape[0], 1))
  return filter_window

def fft_2d(x, y, matrix_in, po, op = lambda x : x-np.mean(np.mean(x))):
  nfftx = int(2**round(np.log2(matrix_in.shape[0])))
  nffty = int(2**round(np.log2(matrix_in.shape[1])))
  nfftx *= po.nfft_multiplier
  nffty *= po.nfft_multiplier
  nfft = (nfftx, nffty)
  fft_filter_window = filter_window(matrix_in)
  matrix_out = matrix_in*fft_filter_window
  matrix_out = matrix_out - np.mean(np.mean(matrix_out))
  matrix_out = np.flipud(np.fft.fftshift(np.fft.fft2(matrix_out, nfft)))
 
  x1 = np.linspace(-nfftx/2+1, nfftx/2, nfftx)
  y1 = np.linspace(-nffty/2, nffty/2-1, nffty)

  dx = x[1] - x[0]
  nx = x1.shape[0]
  x1 = x1/nx/dx

  dy = y[1] - y[0]
  ny = y1.shape[0]
  y1 = y1/ny/dy
 
  return (x1, y1, matrix_out)

def fft_1d(x, matrix_in, dim, po):
  nfft = int(2**round(np.log2(matrix_in.shape[dim])))
  nfft *= po.nfft_multiplier
  fft_filter_window = filter_window(matrix_in)
  matrix_out = matrix_in*fft_filter_window
  matrix_out = matrix_out - mean(mean(matrix_out))
  matrix_out = fft.fftshift(np.fft.fft(matrix_out, nfft, dim), dim)
 
  x1 = np.linspace(-nfft/2, nfft/2 -1, nfft)
  dx = x[1] - x[0]
  nx = x1.shape[0]
  x1 = x1/nx/dx

  return (x1, matrix_out)
 
def fft2d_ifft1d(x, y, matrix_in, po, dim = 1):
  nffty = int(2**round(np.log2(matrix_in.shape[1])))
  nfftx = int(2**round(np.log2(matrix_in.shape[0])))
  nfftx *= po.nfft_multiplier
  nffty *= po.nfft_multiplier
  nfft = (nfftx, nffty)
  matrix_out = matrix_in
  matrix_out = matrix_out - np.mean(np.mean(matrix_out))
  matrix_out = np.fft.fft2(matrix_out, nfft)

  x1 = np.linspace(-nfftx/2, nfftx/2-1, nfftx)
  y1 = np.linspace(-nffty/2, nffty/2-1, nffty)
  x2 = np.fft.fftshift(x1)
  y2 = np.fft.fftshift(y1)

  if dim == 1: 
    matrix_out[:, y2 > 0] *= 0
    matrix_out[x2 != 0, :] *= 2.0
    matrix_out = np.fft.ifft(matrix_out, matrix_in.shape[1], 1)
    matrix_out = np.fft.fftshift(matrix_out, 0)
    matrix_out /= matrix_in.shape[0]
    dw = x[1] - x[0]
    nw = x1.shape[0]
    w1 = x1/nw/dw
  elif dim == 0:
    matrix_out[:, y2 != 0] *= 2.0
    matrix_out[x2 > 0, :] *= 0.0
    matrix_out = np.fft.ifft(matrix_out, matrix_in.shape[0], 0)
    matrix_out = np.fft.fftshift(matrix_out, 1)
    matrix_out /= matrix_in.shape[1]
    dw = y[1] - y[0]
    nw = y1.shape[0]
    w1 = y1/nw/dw

  return (w1, matrix_out)

def coherence_loop(fz, x1, signedness=False):
  nfft = len(x1)
  matrix_out = np.zeros((nfft, nfft), dtype=complex)
  for (xi, kxi) in enumerate(x1):
    for (yi, kyi) in enumerate(x1):
      if ((not signedness) & (kyi > kxi)):
        continue
      if ((kxi+kyi >= -nfft/2) & (kxi+kyi <= nfft/2-1)):
        xyi = x1.searchsorted(kxi+kyi)
        matrix_out[xi, yi] = np.abs(np.mean(fz[xi, :]*fz[yi, :]*np.conj(fz[xyi, :])))
  return matrix_out
 
def cohere(x, Z, signedness=False):
  nfft = int(2**round(np.log2(x.shape[0])))
  x1 = np.linspace(-nfft/2, nfft/2-1, nfft)
  fz = np.fft.fft(Z, nfft, 0)
  if (signedness):
    fz = signed_wavenumbers(Z)
  else:
    fz = np.fft.fftshift(fz, 0)
  matrix_out = np.zeros((nfft, nfft), dtype=complex)
  matrix_out = coherence_loop(fz, x1, signedness)
  dx = x[1] - x[0]
  nx = x1.shape[0]
  x1 = x1/nx/dx
  return (x1, matrix_out)

def signed_wavenumbers(Z):
  nfftx = int(2**round(np.log2(Z.shape[0])))
  nffty = int(2**round(np.log2(Z.shape[1])))
  x1 = np.linspace(-nfftx/2, nfftx/2-1, nfftx)
  y1 = np.linspace(-nffty/2, nffty/2-1, nffty)
  x2 = np.fft.fftshift(x1)
  y2 = np.fft.fftshift(y1)

  matrix_out = np.fft.fft2(Z, (nfftx, nffty))
  # this > operator takes into account of minus sign
  # on exp(i(+kx-iwt)) i.e it is flipped
  matrix_out[:, y2 > 0] *= 0
  matrix_out *= 2
  matrix_out[x2 == 0, y2 == 0] *= 0.5
  matrix_out = np.fft.fftshift(np.fft.ifft(matrix_out, nfftx, 1), 0)
  return matrix_out

def power_spectrum(x, z, dim, po):
  (x1, z1) = fft_1d(x, z, dim, po)

  if dim == 0:
    z1 = z1[x1>=0,:]
    x1 = x1[x1>=0]
    z1 = np.sum(z1,0)
    return (x1, np.sum(z, 1))
  if dim == 1:
    z1 = z1[:,x1>=0]
    x1 = x1[x1>=0]
    z1 = np.transpose(np.sum(z1,0))
    return (x1, z1)
  else:
    return (x1, z1)

def windowed_dft_1d_x(matrix_in, x, kwave, nx_window):
  assert(nx_window > 0)
  nx, dx, Lx = get_lx(x)
  matrix_out = np.zeros(matrix_in.shape)
  itmp = np.arange(0, nx_window)
  xtmp = itmp*dx
  eikx = np.exp(-np.complex(0,1)*kwave*xtmp)
  for k in np.arange(0,nx):
    ind = np.int64(np.remainder(k + itmp, nx))
    for c in np.arange(0, matrix_in.shape[1]):
      tmp = np.dot(matrix_in[ind,c], eikx)
      matrix_out[ind, c] += np.abs(np.sum(tmp.sum()))
  return (2*matrix_out/nx_window**2)

def windowed_dft_2d_xt(matrix_in, x, y, kwave, wwave, nx_window, ny_window):
  assert(nx_window > 0)
  assert(ny_window > 0)
  nx, dx, Lx = get_lx(x)
  ny, dy, Ly = get_lx(y)
  matrix_out = np.zeros(np.shape(matrix_in))
  iind = np.arange(0, nx_window)
  vector_x = iind*dx
  vector_y = np.arange(0, ny_window)*dy
  (Y, X) = np.meshgrid(vector_y, vector_x)
  eikxwt = np.exp(-np.complex(0,1)*(kwave*X - wwave*Y))
  for k in np.arange(0, nx):
    xind = np.int64(np.remainder(k + iind, nx))
    for c in range(0, np.int64(np.shape(matrix_in)[1]-ny_window)):
      tmp = np.multiply(matrix_in[xind, c:c+ny_window], eikxwt)
      matrix_out[xind, c:c+ny_window] += np.abs(tmp.sum(axis=(0,1)))
  return (2*matrix_out/(nx_window*ny_window)**2)

def return_physical_parameters(directory_string):
  sdf_handle = SDF(directory_string + "00000.sdf")
  file_handle = sdf_handle.read()
  B0 = sqrt((mean(file_handle['Magnetic Field/Bx']))**2 + 
    (mean(file_handle['Magnetic Field/By']))**2 + 
    (mean(file_handle['Magnetic Field/Bz']))**2)
  B0_vec = np.zeros(3);
  B0_vec[0] = mean(file_handle['Magnetic Field/Bx'])
  B0_vec[1] = mean(file_handle['Magnetic Field/By'])
  B0_vec[2] = mean(file_handle['Magnetic Field/Bz'])
  wca = 2.0*q0*B0/(4.0*1836.0*me)
  tca = 2.0*pi/wca

  n0e = mean(file_handle['Derived/Number_Density/Electron'])
  n0d = mean(file_handle['Derived/Number_Density/' + ion_string_1])
  try:
    n0a = mean(file_handle['Derived/Number_Density/' + ion_string_2])
    n0a = mean(file_handle['Derived/Number_Density/' + ion_string_2])
    n0a = mean(file_handle['Derived/Number_Density/' + ion_string_2])
  except KeyError:
    n0a = 0.0

  mi = mean(file_handle['Particles/Mass/' + ion_string_1])
  Va = B0/sqrt(mu0*n0e*mi)

  return B0, B0_vec, Va, wca, tca, n0e, n0d, n0a

def get_lx(x):
  dx = mean(x[1:] - x[0:-1])
  lx = x.max() - x.min() + dx
  nx = np.size(x)
  assert(np.abs(1-nx*dx/lx) < 1e-14)
  return nx, dx, lx

def get_nx(directory_string):
  sdf_handle = SDF(directory_string + "00000.sdf")
  file_handle = sdf_handle.read()
  x = file_handle['Grid/Grid/X']
  nx, dx, lx = get_lx(x)
  return x, nx, dx, lx

def plot_scatter_3d(x, y, z, c):
  fig = plt.figure()
  plt.cla()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x, y, z, c)
  plt.show()

def particle_parameters(directory_string, species_name, file_number = 0):
  file_name = directory_string + str(file_number).zfill(5) + ".sdf"
  sdf_handle = SDF(file_name)
  file_handle = sdf_handle.read()
  try:
    vx = file_handle['Particles/Vx/' + species_name]
    vy = file_handle['Particles/Vy/' + species_name]
    vz = file_handle['Particles/Vz/' + species_name]
  except KeyError:
    sdf_handle0 = SDF(directory_string + "00000.sdf")
    file_handle0 = sdf_handle0.read()
    m = file_handle0['Particles/Mass/' + species_name]
    vx = file_handle['Particles/Px/' + species_name]/m
    vy = file_handle['Particles/Py/' + species_name]/m
    vz = file_handle['Particles/Pz/' + species_name]/m

  s_min = np.sqrt(np.min(vx**2 + vy**2 + vz**2))
  s_mean = np.sqrt(np.mean(vx**2 + vy**2 + vz**2))
  s_max = np.sqrt(np.max(vx**2 + vy**2 + vz**2))
  return s_mean, s_min, s_max

def xy_to_rt(x,y):
  return (np.sqrt(x**2 + y**2), np.arctan2(y, x))

def particle_plots(directory_string, po, file_number):
  species_name = po.variable
  nt = sdf_counter(directory_string)

  x, nx, dx, lx = get_nx(directory_string)

  B0, B0_vec, Va, wca, tca, n0e, n0d, n0a = return_physical_parameters(directory_string)

  s0, s_min, s_max = particle_parameters(directory_string, species_name)
  exact_file = directory_string + str(file_number).zfill(5) + ".sdf"

  sdf_handle = SDF(exact_file)
  file_handle = sdf_handle.read()
  time = file_handle['Header']['time']
  try:
    vx = file_handle['Particles/Vx/' + species_name]
    vy = file_handle['Particles/Vy/' + species_name]
    vz = file_handle['Particles/Vz/' + species_name]
  except KeyError:
    vx = file_handle['Particles/Px/' + species_name]
    vy = file_handle['Particles/Py/' + species_name]
    vz = file_handle['Particles/Pz/' + species_name]
    sdf_handle0 = SDF(directory_string + "00000.sdf")
    file_handle0 = sdf_handle0.read()
    mi = mean(file_handle0['Particles/Mass/' + species_name])
    vx = vx/mi
    vy = vy/mi
    vz = vz/mi

  try:
    x = file_handle['Grid/Particles/' + species_name + '/X']
    ind = x < 2*lx
    assert(sum(ind) == len(vx))
  except KeyError:
    ind = np.arange(0, len(vx))

  px = file_handle['Particles/Px/' + species_name]
  py = file_handle['Particles/Py/' + species_name]
  pz = file_handle['Particles/Pz/' + species_name]
  w8 = file_handle['Particles/Weight/' + species_name]

  s = np.sqrt(vx**2 + vy**2 + vz**2)
  
  theta = -np.arctan2(B0_vec[2], B0_vec[0])

  vpara = vx*cos(theta) - vz*sin(theta)
  vperp1 = vy
  vperp2 = vx*sin(theta) + vz*cos(theta)

  xstr = r"Position [$V_A/ \tau_{c \alpha}$]"

  x = x[ind]
  vx_norm = vx[ind]
  vy_norm = vy[ind]
  vz_norm = vz[ind]
  vxstr = r"Velocity $x$ $m/s$"
  vystr = r"Velocity $y$ $m/s$"
  vzstr = r"Velocity $z$ $m/s$"
  
  v0, v_min, v_max = particle_parameters(directory_string, species_name)
  vparams1 = particle_parameters(directory_string, species_name, nt-1)
  v01, v_min1, v_max1 = vparams1
  v_min = min(v_min, v_min1)
  v_max = min(v_max, v_max1)

  if (po.units.rfind("Y") > -1):
    vx_norm /= v0
    vy_norm /= v0
    vz_norm /= v0
    vxstr = r"Velocity $x$ $[|v_{\alpha0}|]$"
    vystr = r"Velocity $y$ $[|v_{\alpha0}|]$"
    vzstr = r"Velocity $z$ $[|v_{\alpha0}|]$"
    v_min /= v0
    v_max /= v0
    s /= v0

  polar = False
  if po.plot_types == "vxvy":
    filename = directory_string + species_name + "_VxVy_" + str(file_number) + ".pdf"
    fo = figure_options("Vx Vy ", vxstr, vystr, filename, po.save_display)
    (xr, xt) = xy_to_rt(vx_norm, vy_norm)
    polar = True
  elif po.plot_types == "vxvz":
    filename = directory_string + species_name + "_VxVz_" + str(file_number) + ".pdf"
    fo = figure_options("Vx Vy ", vystr, vzstr, filename, po.save_display)
    (xr, xt) = xy_to_rt(vx_norm, vz_norm)
    polar = True
  elif po.plot_types == "vyvz":
    filename = directory_string + species_name + "_VyVz_" + str(file_number) + ".pdf"
    fo = figure_options("Vy Vz ", vystr, vzstr, filename, po.save_display)
    (xr, xt) = xy_to_rt(vy_norm, vz_norm)
    polar = True
  elif po.plot_types == "gyrodensity":
    filename = directory_string + species_name + "_GyroDensity_" + str(file_number) + ".pdf"
    gyrostr = r'Gyroangle $arctan(v_y/v_x)$ $rad$'
    xstr = r'Spatial Position $\lambda$'
    fo = figure_options("Gyro-Density ", xstr, gyrostr, filename, po.save_display)
    xx = np.remainder(x, lx/120.0)/(lx/120.0)
    (xr, xt) = xy_to_rt(vx_norm, vy_norm)
    w8_multiplier = 1.0
  elif po.plot_types == "gyroenergy":
    filename = directory_string + species_name + "_GyroEnergy_" + str(file_number) + ".pdf"
    gyrostr = r'Gyroangle $arctan(v_y/v_x)$ $rad$'
    xstr = r'Spatial Position $\lambda$'
    fo = figure_options("Gyro-Density ", xstr, gyrostr, filename, po.save_display)
    xx = np.remainder(x, lx/120.0)/(lx/120.0)
    (xr, xt) = xy_to_rt(vx_norm, vy_norm)
    w8_multiplier = s**2
  elif po.plot_types == "vxvyxmodlambda":
    filename = directory_string + species_name + "_VxVyXmodLambda_" + str(file_number) + ".pdf"
    fo = figure_options("Gyro-Density ", vxstr, vystr, filename, po.save_display)
    xnorm = lx/120.0
    #ind = x < 2.0*lx
    #np.random.shuffle(ind)
    ind = np.random.rand(len(x)) > 0.9
    (xr, xt) = xy_to_rt(vx_norm, vy_norm)
    xx = vx_norm[ind]
    yy = vy_norm[ind]
    zz = x[ind]/xnorm
    zz = np.remainder(zz, 1.0)
    fo.set_method("scatter")
    fo.set_columns(2.5)
    pd = plot_data()
    pd._dim = 3
    pd.add_xyzdata(xx, yy, zz)
    fo.set_plot_data(pd)
    pd1 = fo.plot_data()

  if polar == True:
    nbins = round(len(xr)**(1.0/3.0))
    v_max = 1.25
    r_bins = np.linspace(0.95*v_min, 1.05*v_max, nbins)
    t_bins = np.linspace(-np.pi, np.pi, nbins)
    z_norm, x_edges, y_edges = np.histogram2d(xr, xt, bins=(r_bins,t_bins), weights=w8/xr)
    fo.set_projection('polar')
    fo.set_plot_data(plot_data(3, x_edges, y_edges, z_norm))
    plot_polar(x_edges, y_edges, z_norm, fo, po)
  else:
    if fo.method == "imshow":
      nbins = round(len(xx)**(1.0/3.0))
      x_bins = np.linspace(0, np.max(xx), nbins)
      t_bins = np.linspace(-np.pi, np.pi, nbins)
      z_norm, x_edges, y_edges = np.histogram2d(xx, xt, bins=(x_bins,t_bins), weights=w8*w8_multiplier)
      fo.set_plot_data(plot_data(3, x_edges, y_edges, z_norm))
      plot_2d(x_edges, y_edges, z_norm, fo, po)
    elif fo.method() == "scatter":
      plot_2d(xx, yy, zz, fo, po)



def plot_grid_ic(directory_string, options):
  nt = sdf_counter(directory_string)

  x, nx, dx, lx = get_nx(directory_string)

  B0, B0_vec, Va, wca, tca, n0e, n0d, n0a = return_physical_parameters(directory_string)

  bx = zeros(nx)
  by = zeros(nx)
  bz = zeros(nx)
  ex = zeros(nx)
  ey = zeros(nx)
  ez = zeros(nx)
  jx = zeros(nx)
  jy = zeros(nx)
  jz = zeros(nx)
  na = zeros(nx)
  ne = zeros(nx)
  nd = zeros(nx)

  file_number = "%05d" % 0

  sdf_handle = SDF(directory_string + file_number + ".sdf")
  file_handle = sdf_handle.read()

  jx = file_handle['Current/Jx_averaged']
  jy = file_handle['Current/Jy_averaged']
  jz = file_handle['Current/Jz_averaged']
  ex = file_handle['Electric Field/Ex_averaged']
  ey = file_handle['Electric Field/Ey_averaged']
  ez = file_handle['Electric Field/Ez_averaged']
  bx = file_handle['Magnetic Field/Bx']
  by = file_handle['Magnetic Field/By_averaged']
  bz = file_handle['Magnetic Field/Bz_averaged']
  ne = file_handle['Derived/Number_Density/Electron']
  nd = file_handle['Derived/Number_Density/' + ion_string_1]
  try:
    na = file_handle['Derived/Number_Density/' + ion_string_2]
  except KeyError:
    na = zeros((nx,))

  xstr = r"Position [$V_A/ \tau_{c \alpha}$]"

  print("Magnetic Field X, max = ", np.max(bx), ", min = ",np.min(bx), ", mean = ", np.mean(bx))
  print("Magnetic Field Y, max = ", np.max(by), ", min = ",np.min(by), ", mean = ", np.mean(by))
  print("Magnetic Field Z, max = ", np.max(bz), ", min = ",np.min(bz), ", mean = ", np.mean(bz))
  print("Perturbed magnetic Field energy = ", (np.mean(by**2) + np.mean((bz-np.mean(bz))**2))*0.5/mu0)

  print("Electric Field X, max = ", np.max(ex), ", min = ",np.min(ex), ", mean = ", np.mean(ex))
  print("Electric Field Y, max = ", np.max(ey), ", min = ",np.min(ey), ", mean = ", np.mean(ey))
  print("Electric Field Z, max = ", np.max(ez), ", min = ",np.min(ez), ", mean = ", np.mean(ez))
  print("Perturbed electric Field energy = ", (np.mean(ex**2) + np.mean(ey**2) + np.mean(ez**2))*0.5*epsilon0)

  triplet = (bx, by, bz)
  plot_triplet(x/Va/tca, triplet, xstr, r'Magnetic Fields $T$')
  triplet = (ex, ey, ez)
  plot_triplet(x/Va/tca, triplet, xstr, r'Electric Fields $V/m$')
  triplet = (jx, jy, jz)
  plot_triplet(x/Va/tca, triplet, xstr, r'Currents $C$')
  triplet = (ne, nd, na)
  plot_triplet(x/Va/tca, triplet, xstr, r'Densities $m^{-3}$')

def plot_triplet(x, triplet, xstr, ystr, line_style='-'):
  (a, b, c) = triplet
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.cla()
  plt.plot(x, a,'r'+line_style)
  plt.plot(x, b,'g'+line_style)
  plt.plot(x, c,'b'+line_style)
  plt.xlabel(xstr)
  plt.ylabel(ystr)
  plt.show()

def plot_doublet(x, y, xstr, ystr, line_style='-'):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  plt.cla()
  plt.plot(x, y,'g'+line_style)
  plt.xlabel(xstr)
  plt.ylabel(ystr)
  plt.show()

def plot_energy(directory_string, po, op=lambda x : x, yl1=r'', yl2=r'', label=None):

  if label is None:
    label=directory_string 

  nt = po.number_of_files;
  x, nx, dx, lx = get_nx(directory_string)

  B0, B0_vec, Va, wca, tca, n0e, n0d, n0a = return_physical_parameters(directory_string)

  ee = zeros((nt,))
  eb = zeros((nt,))
  ke = zeros((nt,))
  kd = zeros((nt,))
  ka = zeros((nt,))
  time = zeros((nt,))

  for i in range(0, nt):
    file_number = "%05d" % i

    sdf_handle = SDF(directory_string + file_number + ".sdf")
    file_handle = sdf_handle.read()

    time[i] = file_handle['Header']['time']

    ee[i] = (mean(file_handle['Electric Field/Ex']**2) +
        mean(file_handle['Electric Field/Ey']**2) + 
        mean(file_handle['Electric Field/Ez']**2))*0.5*epsilon0

    eb[i] = (mean(file_handle['Magnetic Field/Bx']**2) +
        mean(file_handle['Magnetic Field/By']**2) + 
        mean(file_handle['Magnetic Field/Bz']**2))*0.5/mu0

    ke[i] = mean(file_handle['Derived/EkBar/Electron'])*n0e
    kd[i] = mean(file_handle['Derived/EkBar/' + ion_string_1])*n0d
    try:
      ka[i] = mean(file_handle['Derived/EkBar/' + ion_string_2])*n0a
    except KeyError:
      ka[i] = 0.0

  et = eb + ee + ke + kd + ka
  print('Energy conservation = ',(et[nt-1] - et[0])/et[0])

  ystr = yl1 + r'Energy density' + yl2 + ' $J/m^3$'
  norm = 1.0
  if (po.units.rfind("V") > -1):
    if (ka[0] > 0):
      ystr = yl1 + r'Energy density' + yl2 + r' [$\mathcal{E}_{\alpha0}$]'
      norm = ka[0]
    else:
      ystr = yl1 + r'Energy density' + yl2 + r' $B_0^2/2\mu_0$'
      norm = 0.5*(B0**2)/mu0
    et /= norm
    eb /= norm
    ee /= norm
    ke /= norm
    kd /= norm
    ka /= norm

  (x_norm, t_norm, labels) = normalise_scales(po.units, x, time, tca*Va, tca)
  (xstr, tstr, kstr, wstr) = labels

  filename = directory_string + "energy_" + str(po.first_file) + "_" + str(po.last_file) 
  fo = figure_options("Energy ", tstr, ystr, filename, po.save_display)

  pd = plot_data(1)
  if po.variable == "None":
    label = ""
    pd.add_xydata(t_norm, op(ee), "Electric" + label)
    pd.add_xydata(t_norm, op(eb), "Magnetic" + label)
    pd.add_xydata(t_norm, op(ke), "Electron" + label)
    pd.add_xydata(t_norm, op(kd), ion_string_1 + label)
    pd.add_xydata(t_norm, op(ka), ion_string_2 + label)
    pd.add_xydata(t_norm, op(et), "Total" + label)
    filename += "_all"
  elif po.variable == "Electron":
    pd.add_xydata(t_norm, op(ke), label)
    filename += "_electron"
  elif po.variable == ion_string_2:
    pd.add_xydata(t_norm, op(ka), label)
    filename += "_ion2"
  elif po.variable == ion_string_1:
    pd.add_xydata(t_norm, op(kd), label)
    filename += "_ion1"
  elif po.variable == ion_string_2 + ion_string_1:
    pd.add_xydata(t_norm, op(ka), label)
    pd.add_xydata(t_norm, op(kd), "")
    filename += "_ion1_ion2"
  elif po.variable == "Ratio":
    pd.add_xydata(t_norm, op(kd)/op(ke), ion_string_1 + "Electron" + label)
    pd.add_xydata(t_norm, op(kd)/op(ee), ion_string_1 + "Electric" + label)
    pd.add_xydata(t_norm, op(kd)/op(eb), ion_string_1 + "Magnetic" + label)
    title = fo.title().strip()
    fo.set_title(title + "_ratios")
    filename += "_ratios"

  filename += ".pdf"
  fo.set_filename(filename)
  fo.set_plot_data(pd)
  fo.set_legend_type(1)

  plot_1d(fo, po)

def plot_momentum(directory_string, po, op=lambda x : x, yl1=r'', yl2=r'', label=None):

  if label is None:
    label=directory_string 

  nt = po.number_of_files;
  x, nx, dx, lx = get_nx(directory_string)

  B0, B0_vec, Va, wca, tca, n0e, n0d, n0a = return_physical_parameters(directory_string)

  mome = zeros((nt,))
  momd = zeros((nt,))
  moma = zeros((nt,))
  poyn = zeros((nt,))
  time = zeros((nt,))

  for i in range(0, nt):
    file_number = "%05d" % i

    sdf_handle = SDF(directory_string + file_number + ".sdf")
    file_handle = sdf_handle.read()
    print file_handle.keys()
    time[i] = file_handle['Header']['time']
    mome[i] = np.mean(file_handle['Grid/x_px/Electron/Px'])
    momd[i] = np.mean(file_handle['Grid/x_px/' + ion_string_1 + "/Px"])
    ey = file_handle['Electric Field/Ey']
    ez = file_handle['Electric Field/Ez']
    by = file_handle['Magnetic Field/By']
    bz = file_handle['Magnetic Field/Bz']
    poyn[i] = np.mean(ey*bz-ez*by)/c0
    try:
      momd[i] = np.mean(file_handle['Grid/x_px/' + ion_string_2 + "/Px"])
    except KeyError:
      moma[i] = 0.0

  ystr = yl1 + r'Momentum density' + yl2 + ' $kg m^-2 s^-1$'
  momt = mome+momd+moma

  (x_norm, t_norm, labels) = normalise_scales(po.units, x, time, tca*Va, tca)
  (xstr, tstr, kstr, wstr) = labels

  filename = directory_string + "momentum_" + str(po.first_file) + "_" + str(po.last_file) 
  fo = figure_options("Momentum ", tstr, ystr, filename, po.save_display)

  pd = plot_data(1)
  if po.variable == "None":
    #pd.add_xydata(t_norm, op(mome), "Electron" + label)
    #pd.add_xydata(t_norm, op(momd), ion_string_1 + label)
    pd.add_xydata(t_norm, op(moma), ion_string_2 + label)
    #pd.add_xydata(t_norm, op(poyn), "Poynting" + label)
    #pd.add_xydata(t_norm, op(momt), "Total" + label)
    filename += "_all"
  elif po.variable == "Electron":
    pd.add_xydata(t_norm, op(mome), label)
    filename += "_electron"
  elif po.variable == ion_string_2:
    pd.add_xydata(t_norm, op(moma), label)
    filename += "_ion2"
  elif po.variable == ion_string_1:
    pd.add_xydata(t_norm, op(momd), label)
    filename += "_ion1"
  elif po.variable == ion_string_2 + ion_string_1:
    pd.add_xydata(t_norm, op(moma), label)
    pd.add_xydata(t_norm, op(momd), "")
    filename += "_ion1_ion2"
  elif po.variable == "poynting":
    pd.add_xydata(t_norm, op(poyn), label)
    filename += "_poynting"

  filename += ".pdf"
  fo.set_filename(filename)
  fo.set_plot_data(pd)
  fo.set_legend_type(1)

  plot_1d(fo, po)

def get_length(x):
  return (x.max() - x.min() + x[1] - x[0])

def parse_inputs(directory_string, plot_opts):

  variables = plot_opts.variable.split(",")
  plot_types = plot_opts.plot_types.split(",")
 
  var_strs = list()
  for variable in variables:
    var_str = ""
    if (variable.rfind("None") > -1):
      continue
    elif (variable.rfind("J") > -1):
      var_str = "Current/" + variable + "_averaged"
    elif (variable.rfind("E") > -1):
      var_str = "Electric Field/" + variable + "_averaged"
    elif (variable.rfind("B") > -1):
      var_str = "Magnetic Field/" + variable + "_averaged"
    elif (variable.rfind("Ne") > -1):
      variable = "Electron"
      var_str = "Derived/Number_Density/" + variable
    elif (variable.rfind("Nd") > -1):
      variable = ion_string_1
      var_str = "Derived/Number_Density/" + variable
    elif (variable.rfind("Na") > -1):
      variable = ion_string_2
      var_str = "Derived/Number_Density/" + variable
    elif (variable.rfind("Ke") > -1):
      variable = "Electron"
      var_str = "Derived/EkBar/" + variable
    elif (variable.rfind("Kd") > -1):
      variable = ion_string_1
      var_str = "Derived/EkBar/" + variable
    elif (variable.rfind("Ka") > -1):
      variable = ion_string_2
      var_str = "Derived/EkBar/" + variable
    elif (variable.rfind("Q") > -1):
      variable = "Charge_Density"
      var_str = "Derived/Charge_Density"
    elif (variable.rfind("Pe") > -1):
      variable = "Electron"
      var_str = "dist_fn/x_px/" + variable
    elif (variable.rfind("Pd") > -1):
      variable = ion_string_1
      var_str = "dist_fn/x_px/" + variable
    elif (variable.rfind("Pa") > -1):
      variable = ion_string_2
      var_str = "dist_fn/x_px/" + variable

    var_strs.append(var_str)
  if len(var_strs) == 0:
    return

  for plot_type in plot_types:
    var_tuple = (variable, var_strs)
    plot_tuple = (plot_opts.units, 
                  plot_opts.save_display,
                  plot_opts.number_of_files, 
                  plot_opts.first_file, 
                  plot_opts.last_file)
    grid_plots(directory_string, 
               plot_type, 
               var_tuple, 
               plot_tuple,
               plot_opts)

def get_2d_data(directory_string, var_strs, nt, first_file):
  x, nx, dx, lx = get_nx(directory_string)
  data = list()
  for var_str in var_strs:
    time = zeros((nt,))
    datum = zeros((nx, nt))
    warning = False
    warning_count = 0
    for i in range(0, nt):
      sdf_handle = SDF(directory_string + "%05d" % (i + first_file) + ".sdf")
      file_handle = sdf_handle.read()
      time[i] = file_handle['Header']['time']
      try:
        datumum = np.squeeze(file_handle[var_str])
        len_datum = len(np.shape(datumum))
        if len_datum == 1:
          datum[:,i] = datumum
        else:
          datum[:,i] = np.sum(datumum, tuple(range(1, len_datum)))
      except KeyError:
        warning = True
        warning_count += 1
        datum[:,i] = zeros((nx,))
    if warning:
      print 'Data ', var_str, 'not found in SDF ', warning_count, ' times'
    else:
      print 'Data ', var_str, 'successfully found in SDF'
    data.append(datum)

  if len(data)==1:
    return (time, data[0])
  else:
    return (time, data)

def normalise_scales(units, x, time, x_norm, time_norm):
  nx = x.shape[0]
  nt = time.shape[0]
  x_new = np.zeros((nx,))
  t_new = np.zeros((nt,))
  # the defaults
  x_new = x/get_length(x)
  t_new = time/get_length(time)
  kstr = r"Wavenumber "
  wstr = r"Frequency "
  xstr = r"Position "
  tstr = r"Time "
  # now change them if desired
  if ((units.rfind("Y") > -1) or (units.rfind("V") > -1)):
    x_new = x/x_norm
    t_new = time/time_norm
    kstr = r"Wavenumber [$ \omega_{c \alpha} / V_A$]"
    wstr = r"Frequency [$ \omega_{c \alpha}$]"
    xstr = r"Position [$V_A \tau_{c \alpha}$]"
    tstr = r"Time [$ \tau_{c \alpha}$]"
  elif (units.rfind("W") > -1):
    x_new = x/x_norm
    t_new = time/time_norm
    kstr = r"Wavenumber [$ \omega_{c \alpha} / V_A$]"
    wstr = r"Frequency [$ \omega_{c \alpha}$]"
    xstr = r"Position [$V_A / \omega_{c \alpha}$]"
    tstr = r"Time [$ \tau_{c \alpha}$]"
  elif (units.rfind("K") > -1):
    x_new = x/x_norm
    t_new = time/time_norm
    kstr = r"Wavenumber [$ \omega_{c\alpha} / V_A$]"
    wstr = r"Frequency [$ \omega_{c \alpha}$]"
    xstr = r"Position [$V_A \omega_{c \alpha}$]"
    tstr = r"Time [$ \tau_{c \alpha}$]"
  elif (units.rfind("T") > -1):
    t_new = time/time_norm
    wstr = r"Frequency [$ \omega_{c \alpha}$]"
    tstr = r"Time [$ \tau_{c \alpha}$]"
  elif (units.rfind("X") > -1):
    x_new = x/x_norm
    kstr = r"Wavenumber [$ \omega_{c \alpha}/V_A$]"
    xstr = r"Position [$V_A/ \omega_{c \alpha}$]"

  labels = (xstr, tstr, kstr, wstr) 

  return (x_new, t_new, labels)

def normalise_data(datum, var_str):
  if (var_str.find('E') > -1):
    datum = datum * np.sqrt(2.0*epsilon0)
  elif (var_str.find('B') > -1):
    datum = datum * np.sqrt(2.0/mu0)
  return datum

def grid_plots(directory_string, plot_type, var_tuple, plot_tuple, po):

  (variable, var_str) = var_tuple
  (units, save_display, nt, first_file, last_file) = plot_tuple

  x, nx, dx, lx = get_nx(directory_string)

  B0, B0_vec, Va, wca, tca, n0e, n0d, n0a = return_physical_parameters(directory_string)

  (time, data) = get_2d_data(directory_string, var_str, nt, first_file)

  (x_norm, t_norm, labels) = normalise_scales(units, x, time, tca*Va, tca)
  (xstr, tstr, kstr, wstr) = labels
   
  filename = directory_string + variable + "_" + plot_type + "_" + \
  str(first_file) + "_" + str(last_file) + ".pdf"
  if (plot_type == "xt"): 
    fo = figure_options("Time-Space", xstr, tstr, filename, save_display)
    plot_2d(x_norm, t_norm, data, fo, po)
  elif (plot_type == "wk"): 
    fo = figure_options("Frequency-Wavenumber", kstr, wstr, filename, save_display)
    (x1, y1, z1) = fft_2d(x_norm, t_norm, data - np.mean(data[:,0]), po)
    plot_2d(x1, y1, np.log10(np.abs(z1)), fo, po)
  elif (plot_type == "tk"): 
    fo = figure_options("Time-Wavenumber", kstr, tstr, filename, save_display)
    (x1, z1) = fft2d_ifft1d(x_norm, t_norm, data - np.mean(data[:,0]), po)
    plot_2d(x1, t_norm, np.log10(np.abs(z1)), fo, po)
  elif (plot_type == "kxt"): 
    nwave = 120
    fx = nwave*2.0*np.pi/get_length(x_norm)
    wx = np.ceil(nx/nwave)
    tmp = windowed_dft_1d_x(data, x_norm, fx, wx)
    fo = figure_options("Wavenumber Windowed ", xstr, tstr, filename, save_display)
    plot_2d(x_norm, t_norm, np.log10(tmp), fo, po)
  elif (plot_type == "wkxt"): 
    nwave = -120
    fx = nwave*2.0*np.pi/get_length(x_norm)
    wx = np.ceil(nx/np.abs(nwave))
    mwave = 179 
    ft = mwave*2.0*np.pi/get_length(t_norm)
    wt = np.ceil(nt/np.abs(mwave))
    tmp = windowed_dft_2d_xt(data, x_norm, t_norm, fx, ft, wx, wt)
    fo = figure_options("Wavenumber-Frequency Windowed ", xstr, tstr, filename, save_display)
    for i in np.arange(nt):
      tmp[:,i] -= np.mean(tmp[:,i])
    plot_2d(x_norm, t_norm, tmp, fo, po)
  elif (plot_type == 'w'):
    data = normalise_data(data, var_str[0])
    fo = figure_options("Power Frequency", wstr, r"Power ", filename, save_display)
    (x1, z1) = fft2d_ifft1d(x_norm, t_norm, data - np.mean(data[:,0]), po, 0)
    z1 = np.sum(z1*np.conj(z1), 0)
    pd = plot_data(1)
    pd.add_xydata(x1, np.log10(np.abs(z1)), fo.filename())
    fo.set_plot_data(pd)
    plot_1d(fo, po)
  elif (plot_type == 'c'):
    fo = figure_options("Coherence", kstr, kstr, filename, save_display)
    (x1, z1) = cohere(x_norm, data - np.mean(data[:,0]))
    plot_2d(x1, x1, np.log10(np.abs(z1)), fo, po)
  elif (plot_type == 'sc'):
    fo = figure_options("Coherence", kstr, kstr, filename, save_display)
    (x1, z1) = cohere(x_norm, data - np.mean(data[:,0]), True)
    plot_2d(x1, x1, np.log10(np.abs(z1)), fo, po)
  elif (plot_type == 'nsc'):
    fo = figure_options("Coherence", kstr, kstr, filename, save_display)
    (x1, z1) = cohere(x_norm, data - np.mean(data[:,0]), True)
    z1 = np.abs(z1)
    z1 /= np.max(np.max(z1))
    plot_2d(x1, x1, np.log10(z1), fo, po)
  elif (plot_type == 'nsc_diff'):
    fo = figure_options("Coherence", kstr, kstr, filename, save_display)
    data_1 = data[:,0:nt/4]
    data_2 = data[:,3*nt/4:nt]
    (x1, z1) = cohere(x_norm, data_1 - np.mean(data_1[:,0]), True)
    (x1, z2) = cohere(x_norm, data_2 - np.mean(data_2[:,0]), True)
    #x1 = np.linspace(-1000, 1000, data.shape[0])
    #z1 = np.random.rand(nx, nx)**6
    #z2 = np.random.rand(nx, nx)**6
    z1 = np.abs(z1)
    z2 = np.abs(z2)
    z1 /= np.max(np.max(z1))
    z2 /= np.max(np.max(z2))
    mn = np.min(np.min(z1))
    mx = np.max(np.max(z1))
    z1 = (z1 - mn + 1.0e-16)/(mx-mn)
    mn = np.min(np.min(z2))
    mx = np.max(np.max(z2))
    z2 = (z2 - mn + 1.0e-16)/(mx-mn)
    z0 = np.log10(z2)-np.log10(z1)
    z0[x1 < -30, :] *= 0 
    z0[x1 > 30, :] *= 0 
    z0[:, x1 < -30] *= 0 
    z0[:, x1 > 30] *= 0 
    z0 /= np.max(np.max(np.abs(z0)))
    plot_2d(x1, x1, z0, fo, po)
  elif (plot_type == "wk_wkxt"): 
    nwave = -120
    fx = nwave*2.0*np.pi/get_length(x_norm)
    wx = np.ceil(nx/np.abs(nwave))
    mwave = 179 
    ft = mwave*2.0*np.pi/get_length(t_norm)
    wt = np.ceil(nt/np.abs(mwave))
    tmp = windowed_dft_2d_xt(data, x_norm, t_norm, fx, ft, wx, wt)
    for i in np.arange(nt):
      tmp[:,i] -= np.mean(tmp[:,i])
    fo = figure_options("Frequency-Wavenumber", kstr, wstr, filename, save_display)
    (x1, y1, z1) = fft_2d(x_norm, t_norm, tmp, po)
    plot_2d(x1, y1, np.log10(np.abs(z1)), fo, po)
  elif (plot_type == "pol"): 
    fo = figure_options("Frequency-Wavenumber", kstr, wstr, filename, save_display)
    datum = data[0]
    datum = normalise_data(datum, var_str[0])
    (x1, y1, z1) = fft_2d(x_norm, t_norm, datum - np.mean(datum[:,0]), po)
    datum = data[1]
    datum = normalise_data(datum, var_str[1])
    (x2, y2, z2) = fft_2d(x_norm, t_norm, datum - np.mean(datum[:,0]), po)
    z3 = np.log10(np.abs(z1/z2))
    z3[x1 == 0,:] = np.zeros((1,nt))
    z3[:,y2 == 0] = np.zeros((nx,1))
    plot_2d(x1, y1, z3, fo, po)


