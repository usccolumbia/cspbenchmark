import pytest 
from agox.utils.batch_analysis import Analysis
from agox.test.test_utils import TemporaryFolder
import os
import shutil

import matplotlib.pyplot as plt
import glob

database_paths = [
    ['datasets/databases/mos2_databases/']
    ]

@pytest.fixture(params=database_paths)
def database_paths(request):
    return request.param

def test_analysis(tmp_path, database_paths):

    absolute_paths = []
    for database_path in database_paths:
        path = os.path.join(tmp_path, 'dataset')
        shutil.copytree(database_path, path)
        absolute_paths.append(path)

    with TemporaryFolder(tmp_path):

        for force_reload in [True, False]:
            analysis = Analysis()
            for path in absolute_paths:
                analysis.add_directory(path, force_reload=False)
            analysis.compile_information()
            analysis.calculate_CDF()

            fig, ax = plt.subplots()
            analysis.plot_CDF(ax)
            analysis.plot_histogram(ax)
            analysis.plot_energy(ax)