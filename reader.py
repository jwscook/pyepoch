#!/opt/local/bin/python2.7

from numpy import *
import numpy as np
import scipy as sp
from sdf import *
from utilities import *
from optparse import OptionParser as op
import operator
from matplotlib import rc
import re

def fix_directory_string(dir_str):
  if ((dir_str[0].rfind(".") != 0) and (dir_str[0].rfind("/") != 0)):
    dir_str = "./" + dir_str
  last_char = dir_str[len(dir_str)-1]
  if (last_char.rfind("/") != 0):
    dir_str = dir_str + "/"

  return dir_str

def main():
  epsilon0 = 8.854187817e-12
  mu0 = pi*4e-7
  q0 = 1.60217657e-19
  me = 9.10938291e-31

  parser = op()
  parser.add_option("-d", "--directory_string", dest="directory_string",
                        help="The directory containing the sdf files", metavar="DIR")

  parser.add_option("-p", "--plot_types", dest="plot_types", default="", 
                        help="The plot types required", metavar="PT")

  parser.add_option("-v", "--variable", dest="variable", default="None", 
                        help="The variable to be plotted", metavar="VAR")

  parser.add_option("-u", "--units", dest="units", default="Y", 
                        help="Use normalised units or integers", metavar="UNITS")

  parser.add_option("-n", "--num_files", dest="number_of_files", default=-1, 
                        help="Number of files", metavar="NFILES", type="int")

  parser.add_option("-w", "--write_save_display", dest="save_display", default=0, 
                        help="Write file to screen or save", metavar="SAVEDISPLAY", type="int")

  parser.add_option("-s", "--stagger", dest="stagger", default=1, 
                        help="Skip every n data point in space and time", 
                        metavar="STAGGER", type="int")

  parser.add_option("-x", "--xlimits", dest="xlimits", default="1.0,-1.0",
                        help="Limits to the plots on x axis", metavar="XLIMITS")
  
  parser.add_option("-y", "--ylimits", dest="ylimits", default="1.0,-1.0",
                        help="Limits to the plots on y axis", metavar="YLIMITS")

  parser.add_option("-m", "--nfft_multipler", dest="nfft_multiplier", default=1, 
                        help="Multiply the number of fourier tansform points by this", 
                        metavar="NFFT_MULT", type="int")

  parser.add_option("-a", "--first_file", dest="first_file", default=0, 
                        help="Integer number of the first file to use",
                        metavar="FIRST_FILE", type="int")

  parser.add_option("-z", "--last_file", dest="last_file", default=-1, 
                        help="Integer number of the last file to use",
                        metavar="LAST_FILE", type="int")

  (options, args) = parser.parse_args()

  options.xlimits = options.xlimits.split(',')
  for i,l in enumerate(options.xlimits):
    options.xlimits[i] = float(l)
  options.ylimits = options.ylimits.split(',')
  for i,l in enumerate(options.ylimits):
    options.ylimits[i] = float(l)
  options.directory_string = options.directory_string.split(',')

  for dir_str in options.directory_string:
    directory_string = fix_directory_string(dir_str)

    nt = sdf_counter(directory_string)
    n_full = 205 #ceil(float(nt-1)/10.0) # 409
    label = r''
    try:
      numbers_in_dir_str = (re.findall('\d', directory_string))
      tmp_value = float(numbers_in_dir_str[0])*10**(-float(numbers_in_dir_str[1]))
      label="{:1.0e}".format(tmp_value)
      label=r'$10^{-' + numbers_in_dir_str[1] + r'}$'
    except:
      label=r'0'

    if (directory_string.find("off") > -0):
      label = "Absent"
    elif (directory_string.find("on") > -0):
      label = "Present"

    if (options.last_file == -1):
      options.last_file = nt - 1
    else:
      assert(options.last_file < nt)
    assert(options.first_file <= options.last_file)

    options.number_of_files = options.last_file - options.first_file + 1

    if (options.plot_types.rfind("x0") > -1): 
      plot_grid_ic(directory_string, options)
    elif (options.plot_types.rfind("v0") > -1): 
      options.plot_types = "vxvy"
      particle_plots(directory_string, options, 0)
      options.plot_types = "vxvz"
      particle_plots(directory_string, options, 0)
      options.plot_types = "vyvz"
      particle_plots(directory_string, options, 0)
    elif (options.plot_types.rfind("vxvy") > -1 or options.plot_types.rfind("gyro") > -1): 
      try:
        particle_plots(directory_string, options, int(nt-1))
      except KeyError:
        pass
      for i in np.arange(0, n_full):
        if i*n_full < nt:
          particle_plots(directory_string, options, int(i*n_full))
    elif (options.plot_types == "e"): 
      fun = lambda x : x - x[0]
      #fun = lambda x : running_mean(x, 21) - x[0]
      plot_energy(directory_string, options, fun, r'$\Delta $', label=label)
    elif (options.plot_types.rfind("lnE") > -1): 
      fun = lambda x : np.log(np.abs(x-x[0]))
      plot_energy(directory_string, options, fun, r'$\mathrm{ln}$ $| \Delta $ ', r'$|$', label)
    elif (options.plot_types.rfind("log10E") > -1): 
      fun = lambda x : np.log10(np.abs(x-x[0]))
      plot_energy(directory_string, options, fun, r'$\mathrm{log_{10}}$ $| \Delta $ ', r'$|$', label)
    elif (options.plot_types == "p"): 
      fun = lambda x : x 
      #fun = lambda x : x - x[0]
      #fun = lambda x : running_mean(x, 21) - x[0]
      plot_momentum(directory_string, options, fun, r'$\Delta $', label=label)

    parse_inputs(directory_string, options)
  os.system('rm *.pkl')

if __name__ == '__main__':   
  main() 

