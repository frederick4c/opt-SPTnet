Segment Large Movies
====================

The segmentation utilities split large experimental movies into SPTnet-sized
HDF5 tiles, run inference on those tiles, and stitch the per-tile predictions
back into full-movie coordinates.

Supported Inputs
----------------

``sptnet-segment`` accepts:

* ``.h5`` and ``.hdf5`` files containing ``timelapsedata`` or ``ims``
* HDF5-backed MATLAB ``.mat`` files
* scipy-readable MATLAB ``.mat`` files
* ``.tif`` and ``.tiff`` stacks

Native Python inputs are interpreted as ``T,Y,X`` for 3D arrays and ``N,T,Y,X``
for 4D arrays. Legacy MATLAB arrays saved as ``H,W,T,N`` should be split with
``--input-axes YXTN``.

Split a Movie
-------------

The recommended tile size for the current SPTnet model is ``30,64,64`` in
``T,Y,X`` order:

.. code-block:: bash

   sptnet-segment RealData/full_realdata.h5 \
       --output-dir RealData/realdata_tiles \
       --block-shape 30 64 64 \
       --overlap 0 0 0 \
       --dtype none

The splitter writes one file per spatial tile. Long movies are split into
temporal clips inside each spatial tile file, so ``timelapsedata`` has shape
``num_temporal_clips,T,Y,X``.

By default, edge tiles are aligned back onto the source movie. For example, an
``89x80`` movie split into ``64x64`` tiles uses starts ``y=0,25`` and ``x=0,16``
instead of creating mostly padded tiles at ``y=64`` and ``x=64``. This gives the
model real image context at the edges and usually reduces edge artefacts. Use
``--no-align-edges`` only when you explicitly need old stride-only behavior.

Run Inference on Tiles
----------------------

The inference command accepts a glob of tile files:

.. code-block:: bash

   sptnet-inference \
       --model-path Trained_models/full_run/trained_model \
       --data "RealData/realdata_tiles/full_realdata_x*.h5" \
       --batch-size 8

On CSD3 with the provided SLURM script:

.. code-block:: bash

   SPT_INFER_DATA="./RealData/realdata_tiles/full_realdata_x*.h5" \
   sbatch slurm/inference_sptnet_csd3.slurm

Result files are written as ``result_<tile name>.h5``. They keep a
``source_file`` attribute pointing back to the source tile; if that path becomes
stale after moving files, the stitcher also checks for the matching tile next to
the ``inference_results`` directory.

Stitch Inference Results
------------------------

After inference, stitch the per-tile predictions into full-movie tracks:

.. code-block:: bash

   sptnet-stitch \
       "RealData/realdata_tiles/inference_results/result_full_realdata_*.h5" \
       --output RealData/stitched_tracks.csv \
       --score-threshold 0.90 \
       --min-track-len 5 \
       --dedup-distance 3.0

The output CSV contains one row per detected point with global ``frame,y,x``
coordinates. Predictions in padded tile regions are discarded before stitching.
Duplicate tracks from overlapping tiles are merged frame-by-frame, keeping the
highest-confidence point in each frame.

``estimation_xy`` from the current SPTnet inference code is interpreted as
``Y,X`` by default. If you need to stitch older files with ``X,Y`` ordering, pass
``--xy-order xy``.

Notebook Visualization
----------------------

Use the stitched visualization helper to inspect results against the full movie
and optional TrackMate labels:

.. code-block:: python

   from IPython.display import HTML, display
   from sptnet.visualization import show_stitched_segmentation_results

   ani, tracks = show_stitched_segmentation_results(
       movie_path="RealData/full_realdata.h5",
       result_pattern="RealData/realdata_tiles/inference_results/result_full_realdata_*.h5",
       threshold=0.90,
       min_track_len=5,
       dedup_distance=3.0,
       show_ground_truth=True,
       return_tracks=True,
   )
   display(HTML(ani.to_jshtml()))

Useful Tuning
-------------

``--overlap``
    Adds intentional tile overlap. This can improve predictions near tile
    borders, but it increases duplicate tracks and inference cost.

``--dedup-distance``
    Controls duplicate merging after stitching. ``2.0`` is conservative,
    ``3.0`` is a good default for overlapping edge-aligned tiles, and ``4.0`` is
    more aggressive.

``--stride``
    Only needed for metadata-free legacy filenames. Modern tile files store
    absolute starts in HDF5 attributes, so no stride argument is needed.
