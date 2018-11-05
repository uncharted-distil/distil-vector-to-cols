import typing
import os
import csv
import collections

import frozendict  # type: ignore
import pandas as pd  # type: ignore

from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from common_primitives import utils

__all__ = ('VectorToColsPrimitive',)


class Hyperparams(hyperparams.Hyperparams):
    vector_col_index = hyperparams.Hyperparameter[typing.Optional[int]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Index of source vector column'
    )
    labels = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Labels for created columns.  If none supplied, labels will auto-generate.'
    )


class VectorToColsPrimitive(transformer.TransformerPrimitiveBase[container.DataFrame,
                                                                 container.DataFrame,
                                                                 Hyperparams]):
    """
    Given the index of a column containing vector data of length N, creates N new columns, each
    containing the data from the corresponding vector element.  Labels for each of the new columns
    can be optionally specified, other wise the name of the original column will be used with a number
    affixed (someVectorCol -> someVectorCol_0, someVectorCol_1).  If the vectors stored in
    the source column of are of differing lengths, the operation will fail.

    TODO:  Should really give the option to fail, truncate, or pad with a value.
    """

    _semantic_types = (
        "https://metadata.datadrivendiscovery.org/types/Location",
        "https://metadata.datadrivendiscovery.org/types/FloatVector"
    )

    __author__ = 'Uncharted Software',
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '1689aafa-16dc-4c55-8ad4-76cadcf46086',
            'version': '0.1.0',
            'name': 'Vector to column convert',
            'python_path': 'd3m.primitives.distil.VectorToCols',
            'keywords': ['vector', 'columns', 'dataframe'],
            'source': {
                'name': 'Uncharted Software',
                'contact': 'mailto:chris.bethune@uncharted.software'
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://gitlab.com/unchartedsoftware/distil-vector-to-cols.git@' +
                               '{git_commit}#egg=distil-vector-to-cols'
                               .format(git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        }
    )

    @classmethod
    def _find_real_vector_column(cls, inputs_metadata: metadata_base.DataMetadata) -> typing.Optional[int]:
        indices = utils.list_columns_with_semantic_types(inputs_metadata, cls._semantic_types)
        return indices[0] if len(indices) > 0 else None

    @classmethod
    def _can_use_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: typing.Optional[int]) -> bool:

        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        if not column_metadata or column_metadata['structural_type'] != str:
            return False

        semantic_types = column_metadata.get('semantic_types', [])

        if set(cls._semantic_types).issubset(semantic_types):
            return True

        return False

    def produce(self, *,
                inputs: container.DataFrame,
                timeout: float = None,
                iterations: int = None) -> base.CallResult[container.DataFrame]:

        # if no column index is supplied use the first real vector column found in the dataset
        vector_idx = self.hyperparams['vector_col_index']
        if vector_idx is None:
            vector_idx = self._find_real_vector_column(inputs.metadata)
        # validate the column
        if not self._can_use_column(inputs.metadata, vector_idx):
            raise exceptions.InvalidArgumentValueError('column idx=' + str(vector_idx) + ' from '
                                                       + str(inputs.columns) + ' does not contain float vectors')
        # flag label generation if none are supplied
        labels = list(self.hyperparams['labels'])
        if labels is None:
            labels = []
        generate_labels = True if labels is None or len(labels) == 0 else False

        # create a dataframe to hold the new columns
        vector_dataframe = container.DataFrame(data=[])

        # loop over elements of the source vector column
        for i, v in enumerate(inputs.iloc[:, vector_idx]):
            elems = v.split(',')
            vector_length = len(elems)
            for j, e in enumerate(elems):
                # initialize columns when processing first row
                if i == 0:
                    # get the name of the source vector column
                    vector_col_metadata = inputs.metadata.query_column(vector_idx)
                    vector_label = vector_col_metadata['name']

                    # create an empty column for each element of the vector
                    if generate_labels:
                        labels.append(vector_label + "_" + str(j))
                    vector_dataframe[labels[j]] = ''

                # write vector elements into each column - force to string as d3m convention is
                # to store data as pandas 'obj' type until explicitly cast
                vector_dataframe.at[i, labels[j]] = str(e.strip())

        # create default d3m metadata structures (rows, columns etc.) and copy the semantic types
        # from the source vector over, replacing FloatVector with Float
        vector_dataframe.metadata = vector_dataframe.metadata.set_for_value(vector_dataframe)
        source_semantic_types = list(inputs.metadata.query_column(vector_idx)['semantic_types'])
        source_semantic_types.remove('https://metadata.datadrivendiscovery.org/types/FloatVector')
        source_semantic_types.append('https://metadata.datadrivendiscovery.org/types/Float')
        for i in range(0, len(labels)):
            vector_dataframe.metadata = vector_dataframe.metadata.\
                update_column(i, {'semantic_types': source_semantic_types})

        output = utils.append_columns(inputs, vector_dataframe)

        # wrap as a D3M container - metadata should be auto generated
        return base.CallResult(output)

    @classmethod
    def can_accept(cls, *,
                   method_name: str,
                   arguments: typing.Dict[str, typing.Union[metadata_base.Metadata, type]],
                   hyperparams: Hyperparams) -> typing.Optional[metadata_base.DataMetadata]:
        output_metadata = super().can_accept(method_name=method_name, arguments=arguments, hyperparams=hyperparams)

        # If structural types didn't match, don't bother.
        if output_metadata is None:
            return None

        if method_name != 'produce':
            return output_metadata

        if 'inputs' not in arguments:
            return output_metadata

        inputs_metadata = typing.cast(metadata_base.DataMetadata, arguments['inputs'])

        # make sure there's a real vector column (search if unspecified)
        vector_col_index = hyperparams['vector_col_index']
        if vector_col_index is not None:
            can_use_column = cls._can_use_column(inputs_metadata, vector_col_index)
            if not can_use_column:
                return None
        else:
            inferred_index = cls._find_real_vector_column(inputs_metadata)
            if inferred_index is None:
                return None

        return inputs_metadata
