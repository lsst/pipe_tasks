import lsst.ip.isr as ipIsr
Isr = ipIsr.IsrTask
root.isr.registry.register("biasIsr", Isr)
root.isr.registry.register("darkIsr", Isr)
root.isr.registry.register("flatIsr", Isr)
root.isr.registry.register("fringeIsr", Isr)
root.isr['biasIsr'].doBias = False
root.isr['biasIsr'].doDark = False
root.isr['biasIsr'].doFlat = False
root.isr['darkIsr'].doBias = True
root.isr['darkIsr'].doDark = False
root.isr['darkIsr'].doFlat = False
root.isr['flatIsr'].doBias = True
root.isr['flatIsr'].doDark = True
root.isr['flatIsr'].doFlat = False
root.isr['fringeIsr'].doBias = True
root.isr['fringeIsr'].doDark = True
root.isr['fringeIsr'].doFlat = True
