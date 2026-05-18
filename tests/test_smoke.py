def test_import_main():
    import app.main

    assert hasattr(app.main, "app")
