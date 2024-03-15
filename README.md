# Sim2Real Unsupervised SAR Target Classification
This is the code address for the paper "Unsupervised Domain Adaptation for SAR Target Classification based on Domain- and Class-level Alignment: from Simulated to Real Data". 
Firstly, you can follow Zhang et.al (Noise-robust target recognition of SAR images based on attribute scattering center matching. Remote Sens. Lett.) to extract the ASC based on sparse representation and use the Hungarian algorithm to match ASC for target classification. Then, you can obtain two prediction results (deep classifier and ASC classifier) andfuse these two predictions to filter the unreliable pseudo labels. The reliable pseudo labels will be put in the "matchv1" filefolder.
Secondly, you can run the train_pseudo_iter.py.
More details will be updated soon. 
## Citation
If you use this dataset and code, please cite the following papers
```
@article{shi2024unsupervised,
  title={Unsupervised domain adaptation for SAR target classification based on domain-and class-level alignment: From simulated to real data},
  author={Shi, Yu and Du, Lan and Li, Chen and Guo, Yuchen and Du, Yuang},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={207},
  pages={1--13},
  year={2024},
  publisher={Elsevier}
}

```
