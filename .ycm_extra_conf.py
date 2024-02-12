def Settings( **kwargs ):
  return {
    'flags': [
        '-x', 'c++',
        '-std=c++2a',
        '-Wall', '-Wextra', '-Werror',
        '-DAI_LOGGING_ENABLED',
        '-Iinclude/',
        '-isystem/home/svalorzen/Projects/eigen-3.4.0-install/include',
        '-isystem/usr/include/python2.7',
        '-isystem./lib/h5cpp/src',
        '-isystem./lib/argparse/include/argparse',
        '-isystem/usr/include/hdf5/serial/'
    ],
  }
