# Custom Scripts
* They were used in a MEG study focusing on the effects of spatial frequency (SF) and color on M170. 

# Processing Workflow and Scripts used
* Digital image processing
1. Adjustment of mean luminance of BSF images: "MakeColorBSFpics.m"
2. Creating spatially-filtered chromatic images: "ImageFiltering_butterworth.m"
3. Creating equiluminant chromatic images: "MakeEquiluminantPics.m"

* Analysis of MEG data
[Behavior Analysis]
1. Analyzing behavioral data
     --> "BehaviorDataAnalysis.py"


[Preprocessing]
2. Oversampled Temporal Projection (OTP) processing of raw data
     --> "OTPprocessing.py"

3. Making forward solution data for ICA processing using mixed source space 
   (surface + Amyg & Hippo)
     --> "ForwardSolution_forPreprocessing.py"

4. Make noise covariance data from system noise data (used in ICA processing)
     --> "MakeNoiseCov_fromSystemNoise.py"

5. Data preprocessing (filtering, ICA for concatenated raw data and epoching)
     --> "Preprocessing.py"

6. ICA processing of epoch data
     --> "ICAprocessingForEpochData.py"

7. Improper trial exclusion by visual inspection
     --> "VisualInspection.py"


[Sensor Data Analysis]
8. Make grand-average datasets of sensor ERF
     --> "MakeGrandAveData_forERFanalysis.py"

9. Plot grand-average ERF datasets
     --> "PlotGrandAveERFdata.py"

10. Statistical analysis (Within-trial comparisons)
     --> "WithinTrialComparison_SensorData.py"

11. Plot results of Within-trial comparisons
     --> "PlotStatResults_WithinTrialComparison.py"

12. Preparation for Sensor-of-interest (SOI) analysis
     --> "MakeERFPeakDatasets.py"

13. SOI statistical analysis
     --> "RM2wayANOVA.R"


[Source Reconstruction]
14. Setting up surface source space & calculate forward solution 
     --> "ForwardSolution_forSurfaceSrcEst.py"

15. Make noise covariance data from participant noise data 
     --> "MakeNoiseCov_fromParticipantNoise.py"

16. Make SourceMorph instance to morph individual brain data to fsaverage
     --> "MakeSrcMorphInstance.py"

17. Source estimation in surface source space using dSPM and ERF data
     --> "SurfSrcEst_dSPM_ERFdata.py"

18. Make decimated fsaverage (6th octahedron) model
     --> "CreateDecimatedFsaverageModel.py"

19. Morph individual brain to decimated fsaverage
     --> "MorphingIndivStcToFsaverage.py"

20. Make grand-average datasets
     --> "MakeGrandAveData_forSourceAnalysis.py"

21. Plot grand-average surface source activity on 3D brain model
     --> "PlotGrandAveSourceData.py"

22. Plot vertices on 3D brain surface
     --> "PlotVertices_onBrainSurf.py"


[Source Statistical Analysis]
23. Repeated measures 2-way ANOVA with a permutation test procedure and TFCE
     --> "ClusterPermTest_2wayANOVA.py"

24. Plot 2-way ANOVA results on 3D brain surface
     --> "Plot2wayANOVAresults_onBrainSurf.py"

25. Plot 2-way ANOVA results with source waveforms
     --> "Plot2wayANOVAresults_withWaveforms.py"

26. 1-way ANOVA with a permutation test procedure and TFCE
     --> "ClusterPermTest_1wayANOVA.py"

27. Plot 1-way ANOVA results on 3D brain surface
     --> "Plot1wayANOVAresults_onBrainSurf.py"

28. Plot 1-way ANOVA results with source waveforms
     --> "Plot1wayANOVAresults_withWaveforms.py"

29. Make mask data for subsequent post hoc paired tests
     --> "MakeMasks_forPosthocPairedTests.py"

30. Post hoc paired tests
     --> "PosthocPairedTests.py"

31. Plot results of post hoc paired tests on 3D brain surface
     --> "PlotPosthocTestsResults_onBrainSurf.py"

32. Statistical analysis (Within-trial comparisons)
     --> "ClusterPermTest_WithinTrialComparisons.py"

33. Plot results of within-trial comparisons on 3D brain surface
     --> "PlotWithinTrialCompResults_onBrainSurf.py"

34. Plot results of within-trial comparisons with source waveforms
     --> "PlotWithinTrialCompResults_withWaveforms.py"
