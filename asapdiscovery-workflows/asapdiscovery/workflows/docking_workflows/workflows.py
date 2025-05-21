"""
Schema for workflows base classes
"""

from typing import Optional

from pydantic.v1 import Field

from asapdiscovery.docking.workflows.docking_workflows import DockingWorkflowInputsBase


class PosteraDockingWorkflowInputs(DockingWorkflowInputsBase):
    postera: bool = Field(
        False, description="Whether to use the Postera database as the query set."
    )

    postera_upload: bool = Field(
        False, description="Whether to upload the results to Postera."
    )
    postera_molset_name: Optional[str] = Field(
        None, description="The name of the molecule set to upload to."
    )
