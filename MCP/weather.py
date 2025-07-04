from mcp.server.fastmcp import FastMCP

mcp = FastMCP('Weather')

@mcp.tool()
async def get_weather(location: str)->str:
    '''Get weather location'''
    return 'Its always humid in delhi'

if __name__=='__main__':
    mcp.run(transport='streamable_http')