from distutils.core import setup

setup(
    name='DistilVectorToCols',
    version='0.1.0',
    description='Converts a vector column to named columns',
    packages=['vectorcols'],
    keywords=['d3m_primitive'],
    install_requires=[
        'pandas == 0.22.0',
        'frozendict==1.2',
        'd3m'
    ],
    entry_points={
        'd3m.primitives': [
            'distil.VectorToCols = vectorcols.vector_to_cols:VectorToColsPrimitive'
        ],
    }
)
