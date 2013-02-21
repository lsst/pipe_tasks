from lsst.pipe.base import Struct

def groupExposures(patchRef, tempExpName, calExpRefList, checkExist=True):
    # compute tempKeyList: a tuple of ID key names in a calExpId that identify a coaddTempExp.
    # You must also specify tract and patch to make a complete coaddTempExp ID.
    butler = patchRef.getButler()
    tempExpKeyList = sorted(set(butler.getKeys(datasetType=tempExpName, level="Ccd")))

    # compute tempExpIdDict, a dict whose:
    # - keys are tuples of coaddTempExp ID values in tempKeyList order
    # - values are a list of calExp data references for calExp that belong in this coaddTempExp
    tempExpIdDict = dict()
    for calExpRef in calExpRefList:
        calExpId = calExpRef.dataId
        if checkExist and not calExpRef.datasetExists("calexp"):
            self.log.warn("Could not find calexp %s; skipping it" % (calExpId,))
            continue

        tempExpIdTuple = tuple(calExpId[key] if key in calExpId else patchRef.dataId[key] for
                               key in tempExpKeyList)
        calExpSubsetRefList = tempExpIdDict.get(tempExpIdTuple)
        if calExpSubsetRefList:
            calExpSubsetRefList.append(calExpRef)
        else:
            tempExpIdDict[tempExpIdTuple] = [calExpRef]

    return Struct(groups=tempExpIdDict, keys=tempExpKeyList)

def getTempExpId(tempExpTuple, keys):
    return dict(zip(keys, tempExpTuple))

def getTempExpRef(butler, tempExpName, tempExpTuple, keys):
    tempExpId = getTempExpId(tempExpTuple, keys)
    return butler.dataRef(datasetType=tempExpName, dataId=tempExpId)
