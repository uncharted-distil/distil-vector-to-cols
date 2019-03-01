"""
   Copyright © 2019 Uncharted Software Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import unittest
from os import path
import csv
import typing

from d3m import container
from d3m.primitives.distil import VectorToCols
from d3m.metadata import base as metadata_base


class VectorToColsPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), 'dataset'))

    _source_semantic_types = set((
        'https://metadata.datadrivendiscovery.org/types/Location',
        'https://metadata.datadrivendiscovery.org/types/FloatVector'
    ))

    _target_semantic_types = set((
        'https://metadata.datadrivendiscovery.org/types/Location',
        'https://metadata.datadrivendiscovery.org/types/Float'
    ))

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

        # verify that the column names were generated as expected
        self.assertListEqual(['d3mIndex', 'lat_lon', 'altitude', 'lat_lon_0', 'lat_lon_1'],
                             list(result_dataframe.columns.values))

        # check that the first row in the dataframe matches the values from the file
        self.assertListEqual(['1', '40.0,116.0', '100', '40.0', '116.0'],
                             list(result_dataframe.iloc[0]))

        self._test_metadata(result_dataframe.metadata, ('lat_lon_0', 'lat_lon_1'))

    def test_with_labels(self) -> None:
        dataframe = self._load_data()

        hyperparams_class = \
            VectorToCols.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults().replace(
            {
                'vector_col_index': 1,
                'labels': ('lat', 'lon')
            }
        )
        vector_to_cols = VectorToCols(hyperparams=hyperparams)
        result_dataframe = vector_to_cols.produce(inputs=dataframe).value

        # verify that we have the expected shape
        self.assertEqual(result_dataframe.shape[0], 4)
        self.assertEqual(result_dataframe.shape[1], 5)

        # verify that the column names were generated as expected
        self.assertListEqual(['d3mIndex', 'lat_lon', 'altitude', 'lat', 'lon'],
                             list(result_dataframe.columns.values))

        # check that the first row in the dataframe matches the values from the file
        self.assertListEqual(['1', '40.0,116.0', '100', '40.0', '116.0'],
                             list(result_dataframe.iloc[0]))

        self._test_metadata(result_dataframe.metadata, ('lat', 'lon'))

    def test_with_inferred_column(self) -> None:
        dataframe = self._load_data()

        hyperparams_class = \
            VectorToCols.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults().replace(
            {
                'labels': ('lat', 'lon')
            }
        )
        vector_to_cols = VectorToCols(hyperparams=hyperparams)
        result_dataframe = vector_to_cols.produce(inputs=dataframe).value

        # verify that we have the expected shape
        self.assertEqual(result_dataframe.shape[0], 4)
        self.assertEqual(result_dataframe.shape[1], 5)

        # verify that the column names were generated as expected
        self.assertListEqual(['d3mIndex', 'lat_lon', 'altitude', 'lat', 'lon'],
                             list(result_dataframe.columns.values))

        # check that the first row in the dataframe matches the values from the file
        self.assertListEqual(['1', '40.0,116.0', '100', '40.0', '116.0'],
                             list(result_dataframe.iloc[0]))

        # test metadata
        self._test_metadata(result_dataframe.metadata, ('lat', 'lon'))

    def test_can_accept_success(self) -> None:
        dataframe = self._load_data()

        # instantiate the primitive and check acceptance
        hyperparams_class = VectorToCols.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults().replace({'vector_col_index': 1})
        vector_to_cols = VectorToCols(hyperparams=hyperparams)
        metadata = vector_to_cols.can_accept(arguments={'inputs': dataframe.metadata},
                                             hyperparams=hyperparams_class.defaults(), method_name='produce')
        self.assertIsNotNone(metadata)

    def test_can_accept_bad_column(self) -> None:
        dataframe = self._load_data()

        # instantiate the primitive and check acceptance
        hyperparams_class = VectorToCols.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        vector_to_cols = VectorToCols(hyperparams=hyperparams_class.defaults())
        hyperparams = hyperparams_class.defaults().replace({'vector_col_index': 2})
        metadata = vector_to_cols.can_accept(arguments={'inputs': dataframe.metadata},
                                             hyperparams=hyperparams, method_name='produce')
        self.assertIsNone(metadata)

    def test_can_accept_inferred_column(self) -> None:
        dataframe = self._load_data()

        # instantiate the primitive and check acceptance
        hyperparams_class = VectorToCols.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        vector_to_cols = VectorToCols(hyperparams=hyperparams_class.defaults())
        hyperparams = hyperparams_class.defaults()
        metadata = vector_to_cols.can_accept(arguments={'inputs': dataframe.metadata},
                                             hyperparams=hyperparams, method_name='produce')
        self.assertIsNotNone(metadata)

    def _load_data(cls) -> container.DataFrame:
        dataset_doc_path = path.join(cls._dataset_path, 'datasetDoc.json')

        # load the dataset and convert resource 0 to a dataframe
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        dataframe = dataset['0']
        dataframe.metadata = dataframe.metadata.set_for_value(dataframe)
        # add attributes that would transerred over from the DatasetToDataframe call
        for s in cls._source_semantic_types:
            dataframe.metadata = dataframe.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 1), s)
        return dataframe

    def _test_metadata(self, metadata: metadata_base.DataMetadata, names: typing.Sequence[str]) -> None:
        self.assertEqual(metadata.query(())['dimension']['length'], 4)
        self.assertEqual(metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length'], 5)

        self.assertEqual(names[0], metadata.query_column(3)['name'])
        self.assertEqual(self._target_semantic_types, set(metadata.query_column(3)['semantic_types']))

        self.assertEqual(names[1], metadata.query_column(4)['name'])
        self.assertEqual(self._target_semantic_types, set(metadata.query_column(4)['semantic_types']))
        return None


if __name__ == '__main__':
    unittest.main()
