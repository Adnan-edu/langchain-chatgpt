from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel


def write_report(filename, html):
    with open(filename, 'w') as f:
        f.write(html)

class WriteReportArgsSchema(BaseModel):
    filename: str
    html: str


#Tool class can only use functions that receive a single argument
#We need to use instance of StructuredTool, if we want that function to receive multiple different arguments

write_report_tool = StructuredTool.from_function(
    name="write_report",
    description="Write an HTML file to disk. Use this tool whenever someone asks for a report.",
    func=write_report,
    args_schema=WriteReportArgsSchema
)