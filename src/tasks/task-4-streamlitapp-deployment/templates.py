import urllib.parse
#this function is supposed to provide page links for the retrieved results on the screen although it doesn't properly work for now
def pagination(total_pages: int, search: str, current_page: int, tags: str) -> str:
    """ HTML scripts to render pagination buttons. """
    # search words and tags
    params = f'?search={urllib.parse.quote(search)}'
    if tags is not None:
        params += f'&tags={tags}'

    # avoid invalid page number (<=0)
    if (current_page - 5) > 0:
        start_from = current_page - 5
    else:
        start_from = 1

    hrefs = []
    if current_page != 1:
        hrefs += [
            f'<a href="{params}&page={1}" target = "_self">&lt&ltFirst </a>',
            f'<a href="{params}&page={current_page - 1}" target = "_self">&ltPrevious</a>',
        ]
        
    for i in range(start_from, min(total_pages + 1, start_from + 10)):
        if i == current_page:
            hrefs.append(f'{current_page}')
        else:
            hrefs.append(f'<a href="{params}&page={i}" target = "_self">{i}</a>')

    if current_page != total_pages:
        hrefs.append(f'<a href="{params}&page={current_page + 1}" target = "_self">Next&gt</a>')

    return '<div>' + '&emsp;'.join(hrefs) + '</div>'


#this embeds css styling into the app
def load_css() -> str:
    """ Return all css styles. """
    common_tag_css = """
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: .15rem .40rem;
                position: relative;
                text-decoration: none;
                font-size: 95%;
                border-radius: 5px;
                margin-right: .5rem;
                margin-top: .4rem;
                margin-bottom: .5rem;
    """
    return f"""
        <style>
            #tags {{
                {common_tag_css}
                color: rgb(88, 88, 88);
                border-width: 0px;
                background-color: rgb(240, 242, 246);
            }}
            #tags:hover {{
                color: black;
                box-shadow: 0px 5px 10px 0px rgba(0,0,0,0.2);
            }}
            #active-tag {{
                {common_tag_css}
                color: rgb(246, 51, 102);
                border-width: 1px;
                border-style: solid;
                border-color: rgb(246, 51, 102);
            }}
            #active-tag:hover {{
                color: black;
                border-color: black;
                background-color: rgb(240, 242, 246);
                box-shadow: 0px 5px 10px 0px rgba(0,0,0,0.2);
            }}
        </style>
    """

#this part displays the number of results retreived for the searched keyword
def number_of_results(total_hits: int, seconds) -> str:
    """ HTML scripts to display number of results and duration. """
    return f"""
        <div style="color:grey;font-size:95%;">
            {total_hits} results in {seconds} seconds
        </div><br>
    """

#this part displays the results retreived for the searched keyword
def search_result(summary: str) -> str:
    """ HTML scripts to display search results. """
    return f"""
      <div style="margin-top: .6rem; margin-bottom: .6rem;">
        <div style="color:grey;font-size:95%;">
            {summary}
        </div>
      </div>
    """

#this part displays tags that finer soft skills which are associated with the results which are the summaried_content column values
#that include the search result
def tag_boxes(search: str, tags: list, active_tag: str) -> str:
    """ HTML scripts to render tag boxes. """
    html = ''
    
    search = urllib.parse.quote(search)
    for tag in tags:
        if tag != active_tag:
            html += f"""
            <a id="tags" href="?search={search}&tags={tag}" target = "_self">
                {tag.replace('-', ' ')}
            </a>
            """
        else:
            html += f"""
            <a id="active-tag" href="?search={search}" target = "_self">
                {tag.replace('-', ' ')}
            </a>
            """
        
    html += '<br><br>'
    return html

def no_result_html() -> str:
    """ """
    return """
        <div style="color:grey;font-size:95%;margin-top:0.5em;">
            No results were found.
        </div><br>
    """


    