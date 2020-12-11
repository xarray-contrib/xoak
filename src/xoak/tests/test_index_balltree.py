import xoak  # noqa:F401


def test_geo_balltree(geo_dataset, geo_indexer):
    # create index
    geo_dataset.xoak.set_index(['lat', 'lon'], 'geo_balltree')

    # select with indexer
    ds_sel = geo_dataset.xoak.sel(lat=geo_indexer.latitude, lon=geo_indexer.longitude)

    # ensure same shape
    assert ds_sel.field.shape == geo_indexer.longitude.shape
