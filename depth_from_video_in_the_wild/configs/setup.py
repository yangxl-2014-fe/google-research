# -*- coding: utf-8 -*-

import sys
import os
import os.path as osp
import yaml
from easydict import EasyDict
import logging
import datetime
import subprocess

cfg_file = osp.join(osp.dirname(__file__), 'depth_from_video_in_the_wild_config.yaml')

if not osp.exists(cfg_file):
    print('cwd:            {}'.format(os.getcwd()))
    print('path not exist: {}'.format(cfg_file))
    raise ValueError


################################################################################
# Do string concatenation in YAML
#   https://stackoverflow.com/questions/5484016/how-can-i-do-string-concatenation-or-string-replacement-in-yaml
################################################################################
# defing custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)

    if len(seq) == 0:
        return ''
    elif len(seq) == 1:
        return str(seq[0])
    elif len(seq) == 2:
        return osp.join(str(seq[0]), str(seq[1]))
    else:
        path = osp.join(str(seq[0]), str(seq[1]))
        for idx in range(2, len(seq)):
            path = osp.join(path, str(seq[idx]))
        return path


# register the tag handler
yaml.add_constructor('!join', join)


################################################################################
# Load YAML
################################################################################
with open(cfg_file) as f_yaml:
    cfg_in_yaml = EasyDict(yaml.load(f_yaml, Loader=yaml.Loader))


################################################################################
# Configuration
################################################################################
class CommonAPI:
    def __init__(self, cfg_dict):
        self.cfg = cfg_dict

    # ======================================================================== #
    # prepare data
    # ======================================================================== #
    def setup_data(self):
        if not osp.exists(self.get_home_dir):
            os.makedirs(self.get_home_dir)

        # config
        if not osp.exists(self.get_config_dir):
            os.makedirs(self.get_config_dir)

        # copy config file
        src_file = cfg_file
        base_name = osp.basename(src_file)
        dst_file = osp.join(self.get_config_dir, base_name)
        str_root, str_ext = osp.splitext(dst_file)
        time_now = '{}'.format(
            datetime.datetime.now().strftime('_%Y-%m-%d'))
        dst_file = str_root + time_now + str_ext

        if sys.platform != 'win32':
            subprocess.call(['cp', src_file, dst_file])
        else:
            os.system('xcopy /s {} {}'.format(src_file, dst_file))
        # ===============================================
        # or:
        #   import shutil
        #   shutil.copyfile("path-to-src", "path-to-dst")
        # ===============================================

        # output
        if not osp.exists(self.get_ou_dir):
            os.makedirs(self.get_ou_dir)

        # log
        if not osp.exists(self.get_log_dir):
            os.makedirs(self.get_log_dir)

        # temp
        if not osp.exists(self.get_temp_dir):
            os.makedirs(self.get_temp_dir)

    # ======================================================================== #
    # < Get Dir >
    # ======================================================================== #
    @property
    def get_home_dir(self):
        return self.cfg.WORKSPACE.HOME

    # ======================================================================== #
    # < Get Config >
    # ======================================================================== #
    @property
    def get_config_dir(self):
        return osp.join(self.get_home_dir, self.cfg.WORKSPACE.CONFIG.DIR)

    @property
    def get_is_log_enabled(self):
        return self.cfg.WORKSPACE.CONFIG.ENABLE_LOG

    @property
    def get_is_release(self):
        return self.cfg.WORKSPACE.CONFIG.RELEASE
    # ======================================================================== #
    # </Get Config >
    # ======================================================================== #

    @property
    def get_ou_dir(self):
        return osp.join(self.get_home_dir, self.cfg.WORKSPACE.OUTPUT.DIR)

    @property
    def get_log_dir(self):
        return osp.join(self.get_home_dir, self.cfg.WORKSPACE.LOG.DIR)

    @property
    def get_log_file(self):
        name_log = osp.join(self.get_log_dir, self.cfg.WORKSPACE.LOG.FILE)
        str_root, str_ext = osp.splitext(name_log)
        time_now = '{}'.format(
            datetime.datetime.now().strftime('_%Y-%m-%d'))
        name_log = str_root + time_now + str_ext
        return name_log

    @property
    def get_temp_dir(self):
        return osp.join(self.get_home_dir, self.cfg.WORKSPACE.TEMP.DATA_TEMP)
    # ======================================================================== #
    # </Get Dir >
    # ======================================================================== #

    # ======================================================================== #
    # Log setting
    # ======================================================================== #
    def setup_log(self):
        if not self.get_is_log_enabled:
            return

        # Initialize logging
        # simple_format = '%(levelname)s >>> %(message)s'
        medium_format = (
            '%(levelname)s : %(filename)s[%(lineno)d]'
            ' >>> %(message)s'
        )
        logging.basicConfig(
            filename=self.get_log_file,
            filemode='w',
            level=logging.INFO,
            format=medium_format
        )
        logging.info('@{} created at {}'.format(
            self.get_log_file,
            datetime.datetime.now())
        )
        print('\n===== log_file: {}\n'.format(self.get_log_file))


################################################################################
# Global config
################################################################################
common_api = CommonAPI(cfg_in_yaml)
common_api.setup_data()
common_api.setup_log()


################################################################################
# Utility
################################################################################
def view_api(obj, brief=True):
    """
    Print api of object.
    """
    logging.warning('view_api( {} )'.format(type(obj)))

    if brief:
        return

    obj_dir = dir(obj)

    api_base = list()
    api_protect = list()
    api_public = list()

    for item_dir in obj_dir:
        if item_dir.startswith('__') or item_dir.endswith('__'):
            api_base.append(item_dir)
        elif item_dir.startswith('_'):
            api_protect.append(item_dir)
        else:
            api_public.append(item_dir)

    enable_sort = False
    if enable_sort:
        api_base.sort()
        api_public.sort()
        api_protect.sort()

    enable_log = True
    if enable_log:
        logging.info('{} {} API {}'.format('=' * 20, type(obj), '=' * 20))

        logging.info('{} public api {}'.format('-' * 10, '-' * 10))
        for item_dir in api_public:
            logging.info('  --> {}'.format(item_dir))

        logging.info('{} protect api {}'.format('-' * 10, '-' * 10))
        for item_dir in api_protect:
            logging.info('  --> {}'.format(item_dir))

        logging.info('{} base api {}'.format('-' * 10, '-' * 10))
        for item_dir in api_base:
            logging.info('  --> {}'.format(item_dir))


def get_specific_files(name_dir, name_ext, with_dir=False):
    """
    Get files with specific extension.

    Parameters
    ----------
    name_dir : str
    name_ext : str
    with_dir : bool

    Returns
    -------
    list of str
    """
    specific_files = list()
    for item_file in os.listdir(name_dir):
        str_base, str_ext = osp.splitext(item_file)
        if str_ext.lower() == name_ext.lower():
            if with_dir:
                specific_files.append(osp.join(name_dir, item_file))
            else:
                specific_files.append(item_file)
    specific_files.sort()
    return specific_files


def get_specific_files_no_tag(name_dir, name_ext, tag_to_exclued,
                              with_dir=False):
    """
    Get files with specific extension and no tag in end of basename.
    Parameters
    ----------
    name_dir : str
    name_ext : str
    tag_to_exclued : str
    with_dir : bool

    Returns
    -------
    list of str
    """
    specific_files = list()
    for item_file in os.listdir(name_dir):
        str_base, str_ext = osp.splitext(item_file)
        if str_ext.lower() == name_ext.lower() \
                and not str_base.endswith(tag_to_exclued):
            if with_dir:
                specific_files.append(osp.join(name_dir, item_file))
            else:
                specific_files.append(item_file)
    specific_files.sort()
    return specific_files


def get_specific_files_with_tag_in_name(name_dir, name_ext, name_tag,
                                        with_dir=False):
    """
    Get files with specific extension and tag in basename.

    Parameters
    ----------
    name_dir : str
    name_ext : str
    name_tag : str
    with_dir : bool

    Returns
    -------
    list of str
    """
    specific_files = list()
    for item_file in os.listdir(name_dir):
        str_base, str_ext = osp.splitext(item_file)
        str_head, str_tail = osp.split(item_file)
        if str_ext.lower() == name_ext.lower() and \
                str_tail.lower().find(name_tag.lower()) != -1:
            if with_dir:
                specific_files.append(osp.join(name_dir, item_file))
            else:
                specific_files.append(item_file)
    specific_files.sort()
    return specific_files


################################################################################
# Print Text with Color
################################################################################
class ColorPrint:
    """
    References
    ----------
    https://stackoverflow.com/a/39452138
    https://i.stack.imgur.com/j7e4i.gif
    """
    def __init__(self):
        pass

    @staticmethod
    def create_color():
        colors = {
            'CEND': '\33[0m',
            'CBOLD': '\33[1m',
            'CITALIC': '\33[3m',
            'CURL': '\33[4m',
            'CBLINK': '\33[5m',
            'CBLINK2': '\33[6m',
            'CSELECTED': '\33[7m',

            'CBLACK': '\33[30m',
            'CRED': '\33[31m',
            'CGREEN': '\33[32m',
            'CYELLOW': '\33[33m',
            'CBLUE': '\33[34m',
            'CVIOLET': '\33[35m',
            'CBEIGE': '\33[36m',
            'CWHITE': '\33[37m',

            'CBLACKBG': '\33[40m',
            'CREDBG': '\33[41m',
            'CGREENBG': '\33[42m',
            'CYELLOWBG': '\33[43m',
            'CBLUEBG': '\33[44m',
            'CVIOLETBG': '\33[45m',
            'CBEIGEBG': '\33[46m',
            'CWHITEBG': '\33[47m',

            'CGREY': '\33[90m',
            'CRED2': '\33[91m',
            'CGREEN2': '\33[92m',
            'CYELLOW2': '\33[93m',
            'CBLUE2': '\33[94m',
            'CVIOLET2': '\33[95m',
            'CBEIGE2': '\33[96m',
            'CWHITE2': '\33[97m',

            'CGREYBG': '\33[100m',
            'CREDBG2': '\33[101m',
            'CGREENBG2': '\33[102m',
            'CYELLOWBG2': '\33[103m',
            'CBLUEBG2': '\33[104m',
            'CVIOLETBG2': '\33[105m',
            'CBEIGEBG2': '\33[106m',
            'CWHITEBG2': '\33[107m'
        }
        return colors

    @staticmethod
    def print_debug(text):
        color_col = ColorPrint.create_color()
        print('{}{}{}'.format(color_col['CBLUE'], text, color_col['CEND']))

    @staticmethod
    def print_info(text):
        color_col = ColorPrint.create_color()
        print('{}{}{}'.format(color_col['CGREEN'], text, color_col['CEND']))

    @staticmethod
    def print_warn(text):
        color_col = ColorPrint.create_color()
        print('{}{}{}'.format(color_col['CYELLOW'], text, color_col['CEND']))

    @staticmethod
    def print_error(text):
        color_col = ColorPrint.create_color()
        print('{}{}{}'.format(color_col['CREDBG'], text, color_col['CBLACKBG']))

    @staticmethod
    def print_color(text, color):
        color_col = ColorPrint.create_color()
        if color.upper() in color_col.keys():
            print('{}{}{}'.format(color_col[color.upper()], text,
                                  color_col['CEND']))
        else:
            print('{}'.format(text))
