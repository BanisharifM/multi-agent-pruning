"""Version information for Multi-Agent LLM Pruning Framework."""

__version__ = "0.2.3"
__author__ = "Mahdi Banisharifdekordi"
__email__ = "msharif@iastate.edu"
__description__ = "Multi-Agent LLM Pruning Framework for Neural Network Compression"
__url__ = "https://github.com/BanisharifM/Multi-Agent-Pruning"

# Version components
VERSION_INFO = {
    'major': 0,
    'minor': 2, 
    'patch': 4,
    'pre_release': None,
    'build': None
}

def get_version_string():
    """Get formatted version string."""
    version = f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"
    
    if VERSION_INFO['pre_release']:
        version += f"-{VERSION_INFO['pre_release']}"
        
    if VERSION_INFO['build']:
        version += f"+{VERSION_INFO['build']}"
        
    return version

def get_version_info():
    """Get detailed version information."""
    return {
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'url': __url__,
        'components': VERSION_INFO
    }

