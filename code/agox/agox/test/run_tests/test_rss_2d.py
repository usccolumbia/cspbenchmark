from agox.test.run_tests.run_utils import agox_test_run

def test_rss(tmp_path, cmd_options):
    mode = 'rss_2d' # This determines the script that is imported.
    agox_test_run(mode, tmp_path, cmd_options)