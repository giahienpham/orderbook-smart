def test_import_version():
    import booksmart

    assert isinstance(booksmart.__version__, str)

