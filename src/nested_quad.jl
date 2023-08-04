measure(segs) = abs(segs[end]-segs[1])
nextatol(atol, segs) = atol/measure(segs)
