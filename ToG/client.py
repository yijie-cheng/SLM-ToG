import typing as tp
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import time
import itertools
import random
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON


def _normalize_qid(entity_url: str) -> str:
    """
    將形如 'http://www.wikidata.org/entity/Qxxx' 的 URI 截斷成 'Qxxx'
    """
    return entity_url.rsplit('/', 1)[-1]


def _normalize_pid(property_url: str) -> str:
    """
    將形如 'http://www.wikidata.org/entity/Pxxx' 的 URI 截斷成 'Pxxx'
    """
    return property_url.rsplit('/', 1)[-1]


class WikidataQueryClient:
    """
    以原本 client.py 的結構和方法命名為基礎，透過 Wikidata 官方 SPARQL Endpoint 實作。
    """
    def __init__(self, url: str, max_retries: int = 3, timeout: int = 60):
        """
        原本是接收 XMLRPC 伺服器的 url，如 "http://localhost:xxxx"。
        為了兼容原本參數，在此直接將 url 存起來，但實際都會連到官方 Endpoint。

        - max_retries: 設定最大重試次數，預設 3 次
        - timeout: 設定 SPARQL 查詢的超時時間（秒），預設 30 秒
        """
        self.url = url
        self.sparql_endpoint = "https://query.wikidata.org/sparql"
        self.max_retries = max_retries
        self.timeout = timeout  # 設定超時時間

    def _run_query(self, query: str) -> tp.Optional[tp.Dict]:
        """
        封裝對 SPARQL Endpoint 的查詢，並加入超時與重試機制。
        """
        for attempt in range(self.max_retries):
            try:
                sparql = SPARQLWrapper(self.sparql_endpoint)
                sparql.setTimeout(self.timeout)  # 設定超時時間
                sparql.setReturnFormat(JSON)
                sparql.setQuery(query)
                
                # 設定 User-Agent，避免被 Wikidata 限制
                sparql.addCustomHttpHeader("User-Agent", "MyWikidataClient/1.0 (https://mywebsite.com)")

                results = sparql.query().convert()
                return results  # 查詢成功時回傳結果

            except Exception as e:
                error_message = str(e)
                
                # 檢查是否是 Rate Limit 或 伺服器超載
                if "429" in error_message or "Too Many Requests" in error_message:
                    print(f"Rate limit reached. Retrying in {2**attempt} seconds... ({attempt+1}/{self.max_retries})")
                    time.sleep(2**attempt + random.uniform(0, 2))  # 指數退避增加等待時間
                
                elif "503" in error_message or "Service Unavailable" in error_message:
                    print(f"Service Unavailable. Retrying in {2**attempt} seconds... ({attempt+1}/{self.max_retries})")
                    time.sleep(2**attempt + random.uniform(0, 2))

                elif "Timeout" in error_message:
                    print(f"SPARQL Query Timeout. Retrying in {2**attempt} seconds... ({attempt+1}/{self.max_retries})")
                    time.sleep(2**attempt + random.uniform(0, 2))

                else:
                    print(f"SPARQL Query Error: {e}")
                    return None  # 其他錯誤則不重試，直接返回 None
        
        print("SPARQL Query failed after multiple retries.")
        return None  # 多次嘗試失敗後返回 None

    def label2qid(self, label: str) -> tp.Union[list, str]:
        """
        以給定文字 (label) 查詢所有符合該 label 的 QID；若無，回傳 'Not Found!'
        原本 server 會回傳 list[Entity] 或 'Not Found!',
        這裡以 list[ { 'qid': 'Qxx', 'label': 'xxx' } ] 或空清單來模擬。
        為簡化，這裡僅回傳 ['Qxx', 'Qyy', ...]。
        """
        query = f"""
        SELECT DISTINCT ?item WHERE {{
          ?item rdfs:label ?lbl .
          FILTER(STR(?lbl) = "{label}")
        }}
        """
        data = self._run_query(query)
        if not data or "results" not in data:
            return "Not Found!"

        qids = []
        for row in data["results"]["bindings"]:
            item_uri = row["item"]["value"]
            qids.append(_normalize_qid(item_uri))

        return qids if qids else "Not Found!"

    def label2pid(self, label: str) -> tp.Union[list, str]:
        """
        查詢所有 label 符合的 Property (PID)；若無，回傳 'Not Found!'.
        """
        query = f"""
        SELECT DISTINCT ?prop WHERE {{
          ?prop rdfs:label ?lbl .
          ?prop rdf:type wikibase:Property .
          FILTER(STR(?lbl) = "{label}")
        }}
        """
        data = self._run_query(query)
        if not data or "results" not in data:
            return "Not Found!"

        pids = []
        for row in data["results"]["bindings"]:
            prop_uri = row["prop"]["value"]
            pids.append(_normalize_pid(prop_uri))

        return pids if pids else "Not Found!"

    def pid2label(self, pid):
        """
        根據 Wikidata 屬性 ID (PID) 獲取標籤。
        輸入: pid (如 'P749')
        輸出: List[Dict[str, str]]，符合 { "pid": "P749", "label": "parent organization" } 格式
        """
        query = f"""
        SELECT ?label WHERE {{
        wd:{pid} rdfs:label ?label.
        FILTER (lang(?label) = "en")
        }}
        """
        results = self._run_query(query)

        if not results or "results" not in results or "bindings" not in results["results"]:
            return []

        return [{"pid": pid, "label": r["label"]["value"]} for r in results["results"]["bindings"]]


    def get_all_relations_of_an_entity(
        self, entity_qid: str
    ) -> tp.Union[dict, str]:
        """
        回傳格式：
        {
        "head": [ { "pid": "Pxx", "label": "PropertyLabel" }, ... ],
        "tail": [ { "pid": "Pyy", "label": "PropertyLabel" }, ... ]
        }
        若查無任何關係則回傳 "Not Found!"。
        """

        query_head = f"""
        SELECT DISTINCT ?propID ?propLabel
        WHERE {{
        wd:{entity_qid} ?p ?o .
        FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
        BIND(STRAFTER(STR(?p), "http://www.wikidata.org/prop/direct/") AS ?propID).

        ?property wikibase:directClaim ?p .
        ?property rdfs:label ?propLabel .
        FILTER(LANG(?propLabel) = "" || LANG(?propLabel) = "en")
        }}
        """

        query_tail = f"""
        SELECT DISTINCT ?propID ?propLabel
        WHERE {{
        ?s ?p wd:{entity_qid} .
        FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
        BIND(STRAFTER(STR(?p), "http://www.wikidata.org/prop/direct/") AS ?propID).

        ?property wikibase:directClaim ?p .
        ?property rdfs:label ?propLabel .
        FILTER(LANG(?propLabel) = "" || LANG(?propLabel) = "en")
        }}
        """

        data_head = self._run_query(query_head)
        data_tail = self._run_query(query_tail)

        if not data_head or "results" not in data_head:
            data_head = {"results": {"bindings": []}}
        if not data_tail or "results" not in data_tail:
            data_tail = {"results": {"bindings": []}}

        head_list = []
        for row in data_head["results"]["bindings"]:
            pid = row["propID"]["value"]
            label = row["propLabel"]["value"]
            head_list.append({"pid": pid, "label": label})

        tail_list = []
        for row in data_tail["results"]["bindings"]:
            pid = row["propID"]["value"]
            label = row["propLabel"]["value"]
            tail_list.append({"pid": pid, "label": label})

        if not head_list and not tail_list:
            return "Not Found!"

        return {
            "head": head_list,
            "tail": tail_list
        }

    def get_tail_entities_given_head_and_relation(
            self, head_qid: str, relation_pid: str
        ) -> dict:
            query_tail = f"""
            SELECT DISTINCT ?entity ?entityLabel WHERE {{
            wd:{head_qid} wdt:{relation_pid} ?entity .
            OPTIONAL {{ ?entity rdfs:label ?entityLabel FILTER(LANG(?entityLabel) = "en") }}
            }}
            """
            tails = []
            data_tail = self._run_query(query_tail)
            if data_tail and "results" in data_tail:
                for row in data_tail["results"]["bindings"]:
                    ent_url = row["entity"]["value"]
                    ent_label = row.get("entityLabel", {}).get("value", "")
                    tails.append({"qid": _normalize_qid(ent_url), "label": ent_label})

            query_head = f"""
            SELECT DISTINCT ?entity ?entityLabel WHERE {{
            ?entity wdt:{relation_pid} wd:{head_qid} .
            OPTIONAL {{ ?entity rdfs:label ?entityLabel FILTER(LANG(?entityLabel) = "en") }}
            }}
            """
            heads = []
            data_head = self._run_query(query_head)
            if data_head and "results" in data_head:
                for row in data_head["results"]["bindings"]:
                    ent_url = row["entity"]["value"]
                    ent_label = row.get("entityLabel", {}).get("value", "")
                    heads.append({"qid": _normalize_qid(ent_url), "label": ent_label})

            return {
                "head": heads,
                "tail": tails
            }

    def get_tail_values_given_head_and_relation(
        self, head_qid: str, relation_pid: str
    ) -> tp.Union[list, str]:
        """
        回傳 literal list，如果沒有則 'Not Found!'.
        """
        query = f"""
        SELECT DISTINCT ?val WHERE {{
          wd:{head_qid} wdt:{relation_pid} ?val .
          FILTER(isLiteral(?val))
        }}
        """
        data = self._run_query(query)
        if not data or "results" not in data:
            return "Not Found!"

        values = []
        for row in data["results"]["bindings"]:
            val = row["val"]["value"]
            values.append(val)

        return values if values else "Not Found!"

    def get_external_id_given_head_and_relation(
        self, head_qid: str, relation_pid: str
    ) -> tp.Union[list, str]:
        """
        與 get_tail_values_given_head_and_relation 類似，但命名區分用於外部 ID。
        """
        query = f"""
        SELECT DISTINCT ?val WHERE {{
          wd:{head_qid} wdt:{relation_pid} ?val .
        }}
        """
        data = self._run_query(query)
        if not data or "results" not in data:
            return "Not Found!"

        # 通常外部ID是 literal，但有些可能是URI，統一先取字串
        values = []
        for row in data["results"]["bindings"]:
            val = row["val"]["value"]
            values.append(val)

        return values if values else "Not Found!"

    def mid2qid(self, mid: str) -> tp.Union[list, str]:
        """
        給定 Freebase MID (如 '/m/0k8z')，查詢對應 Wikidata QID；無則 'Not Found!'.
        即對應 P646 (Freebase ID).
        """
        query = f"""
        SELECT DISTINCT ?item WHERE {{
          ?item wdt:P646 "{mid}" .
        }}
        """
        data = self._run_query(query)
        if not data or "results" not in data:
            return "Not Found!"

        qids = []
        for row in data["results"]["bindings"]:
            item_uri = row["item"]["value"]
            qids.append(_normalize_qid(item_uri))

        return qids if qids else "Not Found!"

    def get_wikipedia_page(self, qid: str, section: str = None) -> str:
        """
        原本是假設 server 會回傳一個 Wikipedia link；這裡改由 SPARQL 取得對應英文維基文章。
        然後用 requests + BeautifulSoup 取得頁面內容。
        如果找不到就回傳 "Not Found!"。
        """
        # 先找出對應的英語維基條目
        query = f"""
        SELECT ?article WHERE {{
          wd:{qid} schema:about ?item ;
                   schema:isPartOf <https://en.wikipedia.org/> .
          BIND(wd:{qid} AS ?item)
        }}
        """
        data = self._run_query(query)
        if not data or "results" not in data or not data["results"]["bindings"]:
            return "Not Found!"

        # 取第一個 article link
        article_url = data["results"]["bindings"][0]["article"]["value"]
        # 發送 requests 取得 HTML
        resp = requests.get(article_url)
        if resp.status_code != 200:
            return f"Failed to retrieve page: {article_url}"

        soup = BeautifulSoup(resp.content, "html.parser")
        content_div = soup.find("div", {"id": "bodyContent"})
        if not content_div:
            return "Not Found!"

        # 移除 script & style
        for script_or_style in content_div.find_all(["script", "style"]):
            script_or_style.decompose()

        if section:
            # 嘗試抓取指定 section
            header = content_div.find(
                lambda tag: tag.name == "h2" and section in tag.get_text()
            )
            if header:
                content = []
                for sibling in header.find_next_siblings():
                    if sibling.name == "h2":
                        break
                    content.append(sibling.get_text())
                return "\n".join(content).strip() if content else ""
            else:
                return f"Section '{section}' not found."

        # 如果沒有指定 section，就回傳文章開頭(在第一個 h2 之前)
        summary_content = []
        for elem in content_div.children:
            # 一旦遇到 <h2> 就停止
            if elem.name == "h2":
                break
            if hasattr(elem, "get_text"):
                summary_content.append(elem.get_text())
        return "\n".join(summary_content).strip()



if __name__ == "__main__":
    client = WikidataQueryClient("dummy_url")
    # print("label2qid('Microsoft') =>", client.label2qid("Microsoft"))
    # print("label2pid('spouse') =>", client.label2pid("spouse"))
    print("pid2label('Q54766749') =>", client.pid2label("Q54766749"))
    # print("qid2label('Q42') =>", client.qid2label("Q42"))
    print("get_all_relations_of_an_entity('Q163727') =>", client.get_all_relations_of_an_entity("Q163727"))
    # print("get_tail_entities_given_head_and_relation('Q163727', 'P69') =>",
    #       client.get_tail_entities_given_head_and_relation("Q163727", "P69"))
    # print("get_tail_values_given_head_and_relation('Q2283', 'P2139') =>",
    #       client.get_tail_values_given_head_and_relation("Q2283", "P2139"))
    # print("get_external_id_given_head_and_relation('Q2283', 'P646') =>",
    #       client.get_external_id_given_head_and_relation("Q2283", "P646"))
    # print("mid2qid('/m/0k8z') =>", client.mid2qid("/m/0k8z"))
    # print("get_wikipedia_page('Q42') =>", client.get_wikipedia_page("Q42"))
