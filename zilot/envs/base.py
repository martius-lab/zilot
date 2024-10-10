import abc
import warnings

from torch import TensorType


class EnvMixin(abc.ABC):
    @abc.abstractmethod
    def reset_to_goal(self, goal: TensorType) -> tuple[TensorType, dict]:
        pass

    """ UTILS """

    @abc.abstractmethod
    def generate_target_goal(self) -> TensorType:
        pass

    """ VISUALIZATION """

    def draw_goals(self, points, colors=None):
        warnings.warn(f"`draw_goals` not implemented for {self.__class__.__name__}")

    def clear_points(self):
        warnings.warn(f"`clear_points` not implemented for {self.__class__.__name__}")

    """ PLOTTING """

    def plot_values(self, ax, v, obs, goal):
        warnings.warn(f"`plot_values` not implemented for {self.__class__.__name__}")

    def plot_policy(self, ax, a, obs, goal):
        warnings.warn(f"`plot_policy` not implemented for {self.__class__.__name__}")
