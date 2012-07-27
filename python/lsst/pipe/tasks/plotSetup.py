"""Setup the matplotlib backend.

Import this module early, to ensure the correct matplotlib backend is chosen.

If you see a warning such as:

 This call to matplotlib.use() has no effect
because the the backend has already been chosen;
matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
or matplotlib.backends is imported for the first time.

then this module is not being imported early enough, or another module is
attempting to redefine the matplotlib backend.  In the latter case, you
could try using the "experimental" switch_backend feature
(http://stackoverflow.com/questions/3285193/how-to-switch-backends-in-matplotlib-python):

import matplotlib.pyplot as pyplot
pyplot.switch_backend('newbackend')

"""

import matplotlib
matplotlib.use('Agg')
