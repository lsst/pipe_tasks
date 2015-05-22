root.defaultMagColumn = 'r'
root.starGalaxyColumn = 'star'
filters = ('u','g','r','i','z')
root.magColumnMap = dict([(f,f) for f in filters])
root.magErrorColumnMap = dict([(f, f + '_err') for f in filters])
root.indexFiles = ['index-photocal-test.fits',] * 20
