import h5py
import numpy as np
import pytest
import tifffile

from sptnet.data.trackmate import (
    combine_tiff_trackmate,
    parse_trackmate_xml,
    tracks_to_dense_positions,
    tracks_to_table,
)


def _write_trackmate_xml(path):
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<Tracks nTracks="2" spaceUnits="pixel" frameInterval="0.5" timeUnits="frame">
  <particle nSpots="2">
    <detection t="1" x="3.5" y="4.5" z="0.0" />
    <detection t="0" x="2.5" y="4.0" z="0.0" />
  </particle>
  <particle nSpots="1">
    <detection t="2" x="8.0" y="9.0" z="1.0" />
  </particle>
</Tracks>
"""
    )


def test_parse_trackmate_xml_sorts_detections_and_flattens_table(tmp_path):
    xml_path = tmp_path / "tracks.xml"
    _write_trackmate_xml(xml_path)

    tracks = parse_trackmate_xml(xml_path)
    table = tracks_to_table(tracks)

    assert [track.length for track in tracks] == [2, 1]
    np.testing.assert_array_equal(tracks[0].detections[:, 0], np.array([0, 1], dtype=np.float32))
    np.testing.assert_allclose(
        table,
        np.array(
            [
                [0, 0, 2.5, 4.0, 0.0],
                [0, 1, 3.5, 4.5, 0.0],
                [1, 2, 8.0, 9.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_tracks_to_dense_positions_uses_nan_for_missing_frames(tmp_path):
    xml_path = tmp_path / "tracks.xml"
    _write_trackmate_xml(xml_path)
    tracks = parse_trackmate_xml(xml_path)

    positions = tracks_to_dense_positions(tracks, num_frames=4)

    assert positions.shape == (2, 4, 2)
    np.testing.assert_allclose(positions[0, 0], np.array([2.5, 4.0], dtype=np.float32))
    assert np.isnan(positions[1, 0]).all()
    np.testing.assert_allclose(positions[1, 2], np.array([8.0, 9.0], dtype=np.float32))


def test_combine_tiff_trackmate_writes_movie_and_tracks(tmp_path):
    tiff_path = tmp_path / "movie.tif"
    xml_path = tmp_path / "tracks.xml"
    output_path = tmp_path / "combined.h5"
    movie = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    tifffile.imwrite(tiff_path, movie)
    _write_trackmate_xml(xml_path)

    result = combine_tiff_trackmate(tiff_path, xml_path, output_path=output_path, dtype="none")

    assert result.movie_shape == (3, 4, 5)
    assert result.num_tracks == 2
    assert result.num_detections == 3
    with h5py.File(output_path, "r") as handle:
        np.testing.assert_array_equal(handle["timelapsedata"][()], movie)
        assert handle["timelapsedata"].attrs["axes"] == "TYX"
        assert handle["trackmate_tracks"].shape == (3, 5)
        assert tuple(value.decode() for value in handle["trackmate_tracks"].attrs["columns"]) == (
            "track_id",
            "frame",
            "x",
            "y",
            "z",
        )
        assert handle["trackmate_positions"].shape == (2, 3, 2)
        np.testing.assert_array_equal(handle["trackmate_lengths"][()], np.array([2, 1], dtype=np.int32))
        assert handle.attrs["format"] == "sptnet-tiff-trackmate"
        assert handle.attrs["frame_interval"] == 0.5


def test_parse_trackmate_xml_rejects_missing_particles(tmp_path):
    xml_path = tmp_path / "empty.xml"
    xml_path.write_text("<TrackMate />")

    with pytest.raises(ValueError, match="particle tracks"):
        parse_trackmate_xml(xml_path)
