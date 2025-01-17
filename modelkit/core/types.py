from types import ModuleType
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

import pydantic
import pydantic.generics

ItemType = TypeVar("ItemType")
ReturnType = TypeVar("ReturnType")


TestReturnType = TypeVar("TestReturnType")
TestItemType = TypeVar("TestItemType")

# This type should be recursive:
# Union[ModuleType, Type, List, str, "LibraryModelsType"]
# but this is not currently supported
LibraryModelsType = Union[ModuleType, Type, List, str]


class TestCases(pydantic.generics.GenericModel, Generic[TestItemType, TestReturnType]):
    item: TestItemType
    result: TestReturnType
    keyword_args: Dict[str, Any] = {}


class ModelTestingConfiguration(
    pydantic.generics.GenericModel, Generic[TestItemType, TestReturnType]
):
    model_keys: Optional[List[str]]
    cases: List[TestCases[TestItemType, TestReturnType]]
