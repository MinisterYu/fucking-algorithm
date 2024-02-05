# coding:utf-8
import os
num = 2
while num < 50:
    curl = """curl 'https://api.qa.climb.tencent.com/cdpb/api/v1/user/management/sync/tag/create' \
  -H 'authority: api.qa.climb.tencent.com' \
  -H 'accept: application/json' \
  -H 'accept-language: zh-CN,zh;q=0.9' \
  -H 'auth-token: 21567e4e63e90a3d2c6500b93f0d133538324dc1d990d94b8537b085f091d70f194ae912a18ee5c26f72697021ca096763d6ed39b071a5c2' \
  -H 'content-type: application/json;charset=UTF-8' \
  -H 'cookie: __root_domain_v=.tencent.com; _qddaz=QD.999593311647780; pgv_pvid=8546825410; _ga=GA1.2.718320191.1689305152; _ga_6WSZ0YS5ZQ=GS1.1.1699176396.2.0.1699176396.0.0.0; _gcl_au=1.1.1596279320.1705562788; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%22100032143407%22%2C%22first_id%22%3A%22189dd65feaa714-060e77d15d7afa8-1a525634-3686400-189dd65feabb6a%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%7D%2C%22identities%22%3A%22eyIkaWRlbnRpdHlfY29va2llX2lkIjoiMTg5ZGQ2NWZlYWE3MTQtMDYwZTc3ZDE1ZDdhZmE4LTFhNTI1NjM0LTM2ODY0MDAtMTg5ZGQ2NWZlYWJiNmEiLCIkaWRlbnRpdHlfbG9naW5faWQiOiIxMDAwMzIxNDM0MDcifQ%3D%3D%22%2C%22history_login_id%22%3A%7B%22name%22%3A%22%24identity_login_id%22%2C%22value%22%3A%22100032143407%22%7D%2C%22%24device_id%22%3A%22189dd65feaa714-060e77d15d7afa8-1a525634-3686400-189dd65feabb6a%22%7D; auth-token=21567e4e63e90a3d2c6500b93f0d133538324dc1d990d94b8537b085f091d70f194ae912a18ee5c26f72697021ca096763d6ed39b071a5c2; JSESSIONID=37F9EC27AEEAF11D42CBD7CBFFF14F35' \
  -H 'origin: https://admin.qa.climb.tencent.com' \
  -H 'referer: https://admin.qa.climb.tencent.com/' \
  -H 'refresh-checksum: 0226aed2be14356233a3645b3a55e78d' \
  -H 'sec-ch-ua: "Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"' \
  -H 'sec-fetch-dest: empty' \
  -H 'sec-fetch-mode: cors' \
  -H 'sec-fetch-site: same-site' \
  -H 'sourcetype: CLIMB' \
  -H 'system-code: CLIMB' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36' \
  --data-raw '{"systemCode":"CLIMB","cipTagId":"35626","cipTagName":"ministeryu","exportTagGroupName":"添加测试任务_"""+str(num)+"""","jobType":"daily","tagType":1,"exportTags":[{"cipTagRuleId":"2143515120529070218","cipTagRuleName":"测试标签_82","customerCount":"0"},{"cipTagRuleId":"2143515125864225269","cipTagRuleName":"测试标签_83","customerCount":"0"},{"cipTagRuleId":"2143515130612177593","cipTagRuleName":"测试标签_84","customerCount":"0"},{"cipTagRuleId":"2143515100631291026","cipTagRuleName":"测试标签_78","customerCount":"0"},{"cipTagRuleId":"2143515144705039626","cipTagRuleName":"测试标签_87","customerCount":"0"}]}' \
  --compressed"""
    os.system(curl)
    num += 1