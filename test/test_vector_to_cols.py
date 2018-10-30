import unittest
from os import path
import csv

from d3m import container
from d3m.primitives.distil import VectorToCols
from d3m.metadata import base as metadata_base


class VectorToColsPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), 'dataset'))

    def test_with_labels(self) -> None:
        dataframe = self._load_data()

        hyperparams_class = \
            VectorToCols.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults().replace(
            {
                'vector_col_index': 1,
                'labels': ('lat', 'lon'),
            }
        )
        vector_to_cols = VectorToCols(hyperparams=hyperparams)
        result_dataframe = vector_to_cols.produce(inputs=dataframe).value

        # verify that we have the expected shape
        self.assertEqual(result_dataframe.shape[0], 4)
        self.assertEqual(result_dataframe.shape[1], 5)

        # # check that column headers are the times
        # self.assertListEqual(times, list(timeseries_dataframe.columns.values))

        # # check that the first row in the dataframe matches the values from the file
        # ts_values = list(timeseries_dataframe.iloc[0])
        # self.assertEqual(len(ts_values), len(values))

    def test_without_labels(self) -> None:
        dataframe = self._load_data()

        hyperparams_class = \
            VectorToCols.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults().replace(
            {
                'vector_col_index': 1
            }
        )
        vector_to_cols = VectorToCols(hyperparams=hyperparams)
        result_dataframe = vector_to_cols.produce(inputs=dataframe).value

        # verify that we have the expected shape
        self.assertEqual(result_dataframe.shape[0], 4)
        self.assertEqual(result_dataframe.shape[1], 5)

        # # check that column headers are the times
        # self.assertListEqual(times, list(timeseries_dataframe.columns.values))

        # # check that the first row in the dataframe matches the values from the file
        # ts_values = list(timeseries_dataframe.iloc[0])
        # self.assertEqual(len(ts_values), len(values))

    # def test_can_accept_success(self) -> None:
    #     dataframe = self._load_timeseries()

    #     # instantiate the primitive and check acceptance
    #     hyperparams_class = TimeSeriesLoader.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    #     ts_reader = TimeSeriesLoader(hyperparams=hyperparams_class.defaults())
    #     metadata = ts_reader.can_accept(arguments={'inputs': dataframe.metadata},
    #                                     hyperparams=hyperparams_class.defaults(), method_name='produce')
    #     self.assertIsNotNone(metadata)

    # def test_can_accept_bad_column(self) -> None:
    #     dataframe = self._load_timeseries()

    #     # instantiate the primitive and check acceptance
    #     hyperparams_class = TimeSeriesLoader.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    #     hyperparams = hyperparams_class.defaults().replace({'file_col_index': 4})
    #     ts_reader = TimeSeriesLoader(hyperparams=hyperparams_class.defaults())
    #     metadata = ts_reader.can_accept(arguments={'inputs': dataframe.metadata},
    #                                     hyperparams=hyperparams, method_name='produce')
    #     self.assertIsNone(metadata)

    # @classmethod
    def _load_data(cls) -> container.DataFrame:
        dataset_doc_path = path.join(cls._dataset_path, 'datasetDoc.json')

        # load the dataset and convert resource 0 to a dataframe
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        dataframe = dataset['0']
        dataframe.metadata = dataframe.metadata.set_for_value(dataframe)
        # add attributes that would transerred over from the DatasetToDataframe call
        dataframe.metadata = dataframe.metadata.\
            add_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                              'https://metadata.datadrivendiscovery.org/types/Location')
        dataframe.metadata = dataframe.metadata.\
            add_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                              'https://metadata.datadrivendiscovery.org/types/FloatVector')
        dataframe.metadata = dataframe.metadata.\
            add_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                              'https://metadata.datadrivendiscovery.org/types/Attribute')
        return dataframe


if __name__ == '__main__':
    unittest.main()
