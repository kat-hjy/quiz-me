"""This module contains the PromptStrategy and PromptContext classes for the prompt strategy pattern."""

from __future__ import annotations
from abc import ABC, abstractmethod
from langgraph.pregel.io import AddableValuesDict
from langgraph.graph.state import CompiledStateGraph


class PromptContext:
    """
    The Context defines the interface of interest to clients.
    """

    def __init__(self, strategy: PromptStrategy) -> None:
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._strategy = strategy

    @property
    def strategy(self) -> PromptStrategy:
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: PromptStrategy) -> None:
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._strategy = strategy

    def generate_response(self, input: dict) -> AddableValuesDict:
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """
        graph: CompiledStateGraph = self._strategy.compile_application()
        input.update({"graph": graph})
        output: AddableValuesDict = self._strategy.generate_output(input)
        return output

    def get_prompt(self) -> None:
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """
        return self._strategy.get_prompt()


class PromptStrategy(ABC):
    """
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def compile_application(self) -> CompiledStateGraph:
        """Compile the application state graph.

        Returns:
            CompiledStateGraph: The compiled application state graph.
        """
        pass

    @abstractmethod
    def generate_output(self, input: dict) -> AddableValuesDict:
        """Generate the output based on the input.

        Args:
            input (dict): The input to generate the output.

        Returns:
            AddableValuesDict: The generated output.
        """
        pass
