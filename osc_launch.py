import sys
try:
    from sklearn.decomposition import PCA
except ImportError:
    # Handle older sklearn versions
    import sklearn.decomposition.pca as pca_module
    sys.modules['sklearn.decomposition._pca'] = pca_module