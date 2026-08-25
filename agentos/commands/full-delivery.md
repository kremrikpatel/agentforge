+++
title = "Full Delivery"
description = "The default workflow: all four stages, analyst to release."
stages = ["analysis", "develop", "test", "deploy"]
bypass_cache = false
+++

Runs the complete team handoff chain: Ana produces the AnalysisContract, Dev
designs against it, Tess tests the design, Dep decides go/no-go. Use this when
nothing else matches -- it is what `/pipeline/run` does without arguments.
