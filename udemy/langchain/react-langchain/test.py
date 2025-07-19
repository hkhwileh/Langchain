from langchain.tools import tool


@tool(description="this function is used to find text length ")
def get_text_length(txt: str) -> int:
    return len(txt)


ValueError(f"Some issue happen when i try to get the text lenght")


tools = [get_text_length]
tool = tools[0]

res = tool.func("dog")
print(res)
