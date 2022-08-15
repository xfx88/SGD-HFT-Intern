import os


class t0_viewer:

    @staticmethod
    def upload(model_name, source_files):
        """
        上传模型信号文件
        :param model_name: 模型名称
        :param source_files: 源文件
        :return: Boolean
        """
        try:
            model_file_name = f"/data/t0_data_view/{model_name}"
            if not os.path.exists(model_file_name):
                os.mkdir(model_file_name)
            os.system(fr"cp -r {source_files}/* {model_file_name}/")
        except Exception as e:
            print(e)
            return False
        return True
