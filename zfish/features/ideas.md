## SpatialData project ideas

### Support for high throughput 3D(T) image data

#### Goal
Represent a multiplexed image dataset in a `SpatialData | Collection[SpatialData]`.  
My PhD dataset can serve as an example:
4 acquisitions * 3 channels (250, 2000, 2000) acquired over ~2 week on a Spinning Disk Microscope. Each field-of-view contains one embryo.

Images:
- raw data (5 TB of unstructured microscopy data).
- compressed (3 TB, 1 `.h5` file per ROI).
- aligned (3 TB, 1 `.h5` file per ROI).

Labels:
- 5 segmented structures (embryo, cell, nucleus, cytoplasm, gene_locus)
- object hierarchy (pd.DataFrame)

Features:
- resources
- feature_type: label, intensity, distance, correlation, neighborhood


#### Layout option:
- `SpatialDataSet` lazy (!) collection of multiple `SpatialData` objects representing a multi-well experiment. 
  - Contains plate & well metadata
- selecting, slicing & subsetting: `SpatialDataSet` -> `SpatialDataSet` | `SpatialData`.
- dataset iterator: `SpatialDataSet` -> `SpatialData`.
- data iterator: `SpatialData` -> `SpatialData`.






### Intensity correction "transformations" & image filtering
- Similar to the existing transformations, Images could be on the fly intensity corrected (i. e. flatfield, z-decay, etc...). Could also double for general image filtering.
  
```python
class IntensityTransform

```

#### Time resolved data
- 



### Open questions:
- Where do channels belong:
  - {'ch0': (Multiscale)SpatialImage[Z, Y, X], 'ch1': (Multiscale)SpatialImage[Z, Y, X]}
  - {'all_channels': (Multiscale)SpatialImage[C, Z, Y, X]}
  - {'cycle0': (Multiscale)SpatialImage[C, Z, Y, X], 'cycle1': (Multiscale)SpatialImage[C, Z, Y, X]}
- Why is there only a single table per `SpatialData` instance? 