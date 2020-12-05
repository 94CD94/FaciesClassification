from cx_Freeze import setup, Executable
import sys
import os

os.environ['TCL_LIBRARY'] = r'C:\Users\ridar\anaconda3\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\ridar\anaconda3\tcl\tk8.6'


base = 'console'

executables = [Executable(r'C:\Users\ridar\Desktop\Tesi_Carniani\Model_evaluation_001.py', base=base)]

build_exe_options = {
'include_files':[r'C:\Users\ridar\Desktop\Tesi_Carniani\Parametri.txt',r'C:\Users\ridar\Desktop\Tesi_Carniani\Pipelines.txt' ],
'includes': ['numpy','seaborn','matplotlib','sklearn','pandas'],
'packages': ['numpy','seaborn'],
'excludes' : [
              'boto.compat.sys',
              'OpenSSL',
              'boto.compat._sre',
              'boto.compat._json',
              'boto.compat._locale',
              'boto.compat._struct',
              'boto.compat.array',
              'prometheus_client',
              'partd',
              'parso',
              'prompt_toolkit',
              'psutils',
              'pygments',
              'notebook',
              'h5py',
              'dask',
              'atomcwrites',
              'babel',
              'et_xmlfiles',
              'tkinter', 'PyQt4.QtSql', 
              'sqlite3', 
              'backports',
              'bokeh',
              'scipy.lib.lapack.flapack',
              'PyQt4.QtNetwork',
              'PyQt4.QtScript',
              'numpy.core._dotblas', 
              'PyQt5',
              'email,'
              'xml',
              'xmlrpc',
              'time',
              'html',
              'imblearn',
              'graphviz',
              'glue',
              'zict',
              'zmq',
              'idna',
              'wsgiref'],
'optimize' : 1
}

setup(
    name = 'modev',
    options = {'build_exe': build_exe_options},
    version = '0.01',
    description = 'any',
    executables = executables
)

input()