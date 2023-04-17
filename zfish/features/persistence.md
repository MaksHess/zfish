# Definition of feature & metadata storage

## Things we would like to store:
* Image (multiscale)
* LabelImage (multiscale)
* LabelObject (as defined by LabelImage & label)
* Feature
* Workflow parameters
* Points
* Bounding box
* Distance matrix 
* KDTree
* Models (i. e. Intensity normalization or cell cycle phase)
* (Multiscale bounding box / Physical bounding box)

### Image and LabelImage
Are stored in `h5` / `zarr` containers (currenlty with custom schema, future OME-Zarr).  
Intensity Image: `UInt16`  
Label Image: `UInt16`

Small scale stuff like flat-field / dark-field stored as `.tiff` (`Float32`)

### LabelObject
Fast access to individual LabelObjects and their relationships via object table:  
`(roi: str, structure: str, label: UInt16, [level: int] ,  *unique(structure): UInt16, *bbx: UInt16, *centroid: Float32)`

### Bounding box
Defined in pixel coordinates, only valid together with roi & level information.  
`(roi: str, level: int, bbx_z_lower: Int32, bbx_z_upper: Int32, bbx_y_lower: Int32, bbx_y_upper: Int32, bbx_x_lower: Int32, bbx_x_upper: Int32)`  
Alternative represencation as `tuple[slice]` not practical for serialization.

### Point
Defined in physical coordinates!  
`(roi: str, id: UInt32, z: float, y: float, x: float)`

### Feature
TBA: Name contain all relevant processing information.

### Distance matrix
Computed on-the-fly for now.

### KD Tree
Computed on-the-fly for now.

### Workflow parameters
YAML / JSON file.

### Models
* Intensity decay models:
    - One (or more?) per intensity image.
    - Store: `(int_img, *params)`.

* Cell cycle phase models:
    - Many, especially when parameter tuning
    - Store: All relevant parameters to generate the model (features, preprocessing, normalization, parameters, etc...)

## AnnData vs. Parquet
Parquet seems like a great choice for feature storage (most of the things ⬆️ can be readily expressed in columns). Built on top of apache arrow it is battle-tested, has strict types, allows reading of individual columns or row-chunks.  
Unfortunately the OME-zarr specification decided on AnnData so we will try to adhere to their standard (or at least stay compatible).