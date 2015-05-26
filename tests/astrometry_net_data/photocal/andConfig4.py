root.defaultMagColumn = 'r'
root.starGalaxyColumn = 'star'
#root.variableColumn = 'variable'
filters = ('u','g','r','i','z')
root.magColumnMap = dict([(f,f) for f in filters])
root.magErrorColumnMap = dict([(f, f + '_err') for f in filters])
root.multiIndexFiles = [ ['index-photocal-test.fits',
                          'index-photocal-test.fits',
                          'mindex-photocal-test-4.fits' ], ]

