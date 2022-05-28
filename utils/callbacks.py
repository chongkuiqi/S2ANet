# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Callback utils
"""


class Callbacks:
    """"
    Handles all registered callbacks for YOLOv5 Hooks
    """

    def __init__(self):
        # Define the available callbacks
        # 定义回调函数类型
        # 回调：回调函数是你写的，即你知道要怎样去作相关的处理，但不知道什么时候去做，因此把回调函数交给第三方，让它去调用回调函数然后返回结果给你
        self._callbacks = {
            'on_pretrain_routine_start': [],
            'on_pretrain_routine_end': [],

            'on_train_start': [],
            'on_train_epoch_start': [],
            'on_train_batch_start': [],
            'optimizer_step': [],
            'on_before_zero_grad': [],
            'on_train_batch_end': [],
            'on_train_epoch_end': [],

            'on_val_start': [],
            'on_val_batch_start': [],
            'on_val_image_end': [],
            'on_val_batch_end': [],
            'on_val_end': [],

            'on_fit_epoch_end': [],  # fit = train + val
            'on_model_save': [],
            'on_train_end': [],
            'on_params_update': [],
            'teardown': [],
        }

        self.stop_training = False  # set True to interrupt training

    # 登记回调函数的动作
    def register_action(self, hook, name='', callback=None):
        """
        Register a new action to a callback hook

        Args:
            hook        The callback hook name to register the action to
            name        The name of the action for later reference
            callback    The callback to fire
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        # callable(object)是python内置函数，用于检查一个object是否是可调用的。
        # 如果返回 True，仍然可能调用失败；但如果返回 False，绝对无法调用。
        # 对于函数、方法、lambda 函式、 类，以及实现了 __call__ 方法的类实例, 它都返回 True。
        assert callable(callback), f"callback '{callback}' is not callable"
        # 增加一个列表元素，元素为字典，包括动作的名称，以及具体的callback函数
        self._callbacks[hook].append({'name': name, 'callback': callback})

    def get_registered_actions(self, hook=None):
        """"
        Returns all the registered actions by callback hook

        Args:
            hook The name of the hook to check, defaults to all
        """
        # 找到某个具体的hook
        if hook:
            return self._callbacks[hook]
        else:
            return self._callbacks

    def run(self, hook, *args, **kwargs):
        """
        Loop through the registered actions and fire all callbacks

        Args:
            hook The name of the hook to check, defaults to all
            args Arguments to receive from YOLOv5
            kwargs Keyword Arguments to receive from YOLOv5
        """

        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"

        for logger in self._callbacks[hook]:
            logger['callback'](*args, **kwargs)
