import numpy as np

from src.models.base import ModelBase


class TestBase:

    def test_init(self, tmp_path):
        ModelBase(tmp_path)
        assert (tmp_path / 'models').exists(), f'Models dir not made!'

    def test_evaluate(self, tmp_path, monkeypatch, capsys):

        def mockreturn(self):

            y = np.array([1, 1, 1, 1, 1])

            test_arrays = {'hello': {'y': y}}
            preds_arrays = {'hello': y}

            return test_arrays, preds_arrays

        monkeypatch.setattr(ModelBase, 'predict', mockreturn)

        base = ModelBase(tmp_path)
        base.evaluate(save_results=False, save_preds=False)

        captured = capsys.readouterr()
        expected_stdout = 'RMSE: 0.0'
        assert expected_stdout in captured.out, \
            f'Expected stdout to be {expected_stdout}, got {captured.out}'
