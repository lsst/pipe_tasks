root.defaultMagColumn = 'r'
root.starGalaxyColumn = 'starnotgal'
#root.variableColumn = 'variable'
filters = ('u', 'g', 'r', 'i', 'z')
root.magColumnMap = dict([(f, f) for f in filters])
root.magErrorColumnMap = dict([(f, f + '_err') for f in filters])
root.indexFiles = ['index-tst2155-00.fits']
