package com.example.demo.service;

import java.lang.*;
import com.alibaba.fastjson.JSON;
import com.example.demo.pojo.Content;
import com.example.demo.pojo.QApair;
import com.example.demo.utils.JsonParseUtil;
import de.l3s.boilerpipe.BoilerpipeProcessingException;
import org.elasticsearch.action.bulk.BulkRequest;
import org.elasticsearch.action.bulk.BulkResponse;
import org.elasticsearch.action.index.IndexRequest;
import org.elasticsearch.action.search.SearchRequest;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.client.RequestOptions;
import org.elasticsearch.common.unit.TimeValue;
import org.elasticsearch.common.xcontent.XContentType;
import org.elasticsearch.index.query.BoolQueryBuilder;
import org.elasticsearch.index.query.MatchQueryBuilder;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.index.query.TermQueryBuilder;
import org.elasticsearch.search.SearchHit;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;
import org.elasticsearch.client.RestHighLevelClient;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

import com.example.demo.utils.HtmlParseUtil;
import org.xml.sax.SAXException;

@Service
public class ContentService {

    //将客户端注入
    @Autowired
    @Qualifier("restHighLevelClient")
    private RestHighLevelClient client;

    //1、解析数据放到 es 中
    public boolean parseContent(String keyword) throws IOException, InterruptedException {
        List<QApair> contents = new HtmlParseUtil().parseResults(keyword);
        //把查询的数据放入 es 中
        BulkRequest request = new BulkRequest();
        request.timeout("2m");

        for (int i = 0; i < contents.size(); i++) {
            request.add(
                    new IndexRequest("question_answer")
//                    new IndexRequest("bing_results")
                            .source(JSON.toJSONString(contents.get(i)), XContentType.JSON));
        }
        BulkResponse bulk = client.bulk(request, RequestOptions.DEFAULT);
        return !bulk.hasFailures();
    }

    //2、获取这些数据实现基本的搜索功能
    public List<Map<String, Object>> searchPage(String keyword, int pageNo, int pageSize) throws IOException {
        //keyword="机器学习";
       // keyword=keyword.getBytes("UTF-8").toString();
        if (pageNo <= 1) {
            pageNo = 1;
        }
        if (pageSize <= 1) {
            pageSize = 1;
        }

        //条件搜索
        SearchRequest searchRequest = new SearchRequest("bing_results");
        SearchSourceBuilder sourceBuilder = new SearchSourceBuilder();

        //分页
        sourceBuilder.from(pageNo).size(pageSize);

        //精准匹配
       // TermQueryBuilder termQuery = QueryBuilders.termQuery("title", keyword);
        MatchQueryBuilder matchQuery = QueryBuilders.matchQuery("title", keyword);


        //sourceBuilder.query(termQuery);
        sourceBuilder.query(matchQuery);
        sourceBuilder.timeout(new TimeValue(60, TimeUnit.SECONDS));
        //执行搜索
        SearchRequest source = searchRequest.source(sourceBuilder);
        SearchResponse searchResponse = client.search(searchRequest, RequestOptions.DEFAULT);
        //解析结果

        List<Map<String, Object>> list = new ArrayList<>();
        for (SearchHit documentFields : searchResponse.getHits().getHits()) {
            list.add(documentFields.getSourceAsMap());
        }
        return list;
    }

    public List<Map<String, Object>> searchQA(String keyword, int pageNo, int pageSize) throws IOException {
        //keyword="机器学习";
        // keyword=keyword.getBytes("UTF-8").toString();
        if (pageNo <= 1) {
            pageNo = 1;
        }
        if (pageSize <= 1) {
            pageSize = 1;
        }

        //条件搜索
        SearchRequest searchRequest = new SearchRequest("question_answer");
        SearchSourceBuilder sourceBuilder = new SearchSourceBuilder();

        //分页
        sourceBuilder.from(pageNo).size(pageSize);

        //精准匹配 --- 不调整排序算法
        // TermQueryBuilder termQuery = QueryBuilders.termQuery("title", keyword);
        //sourceBuilder.query(termQuery);

//        MatchQueryBuilder matchQuery = QueryBuilders.matchQuery("qzh", keyword);
//        sourceBuilder.query(matchQuery);

        //调整排序算法 ---boost
        String[] keyword_buff = keyword.trim().split(" ");
        if(keyword_buff.length<=1){
            MatchQueryBuilder matchQuery = QueryBuilders.matchQuery("question", keyword);
            sourceBuilder.query(matchQuery);
        }
        else{
            MatchQueryBuilder matchQuery1 = QueryBuilders.matchQuery("question", keyword_buff[0]);
            matchQuery1.boost(2);

            String keyword_left=keyword_buff[1];
            for(int i=1;i<keyword_buff.length;i++){
                keyword_left=" "+keyword_buff[i];
            }
            MatchQueryBuilder matchQuery2 = QueryBuilders.matchQuery("question", keyword_left);
            BoolQueryBuilder boolQueryBuilder=QueryBuilders.boolQuery();
            boolQueryBuilder.should(matchQuery1);
            boolQueryBuilder.mustNot(matchQuery2);
            sourceBuilder.query(boolQueryBuilder);
        }



        sourceBuilder.timeout(new TimeValue(60, TimeUnit.SECONDS));
        //执行搜索
        SearchRequest source = searchRequest.source(sourceBuilder);
        SearchResponse searchResponse = client.search(searchRequest, RequestOptions.DEFAULT);
        //解析结果

        List<Map<String, Object>> list = new ArrayList<>();
        for (SearchHit documentFields : searchResponse.getHits().getHits()) {
            list.add(documentFields.getSourceAsMap());
        }
        return list;
    }


    public List<Map<String, Object>> searchAnswer(String id) throws IOException {
        //条件搜索insurance_question
        SearchRequest searchRequest = new SearchRequest("question_answer");
        SearchSourceBuilder sourceBuilder = new SearchSourceBuilder();

        //精准匹配
        TermQueryBuilder termQuery = QueryBuilders.termQuery("id", id);
        //TermQueryBuilder matchQuery = QueryBuilders.termQuery("qid", qid);

        sourceBuilder.query(termQuery);
        //sourceBuilder.query(matchQuery);
        sourceBuilder.timeout(new TimeValue(60, TimeUnit.SECONDS));
        //执行搜索
        SearchRequest source = searchRequest.source(sourceBuilder);
        SearchResponse searchResponse = client.search(searchRequest, RequestOptions.DEFAULT);
        //解析结果

        List<Map<String, Object>> list = new ArrayList<>();
        for (SearchHit documentFields : searchResponse.getHits().getHits()) {
            list.add(documentFields.getSourceAsMap());
        }

        //
        List<Map<String, Object>> list2 = new ArrayList<>();
        String answer="";
        String context="";
        List<String> all_answers= new ArrayList<>();
        String question="";
//        int id="";
        boolean isBing=false;
        if (!list.isEmpty()){
            //条件搜索insurance_answer
            searchRequest = new SearchRequest("question_answer");
            answer= (String) list.get(0).get("answer");
            question= (String) list.get(0).get("question");
//            id=(in) list.get(0).get("id");
            all_answers.addAll((Collection<? extends String>) list.get(0).get("all_answers"));
            isBing=(boolean) list.get(0).get("bing");
            context=(String) list.get(0).get("context");
            //精准匹配
            termQuery = QueryBuilders.termQuery("id", id);
            //TermQueryBuilder matchQuery = QueryBuilders.termQuery("qid", qid);

            sourceBuilder.query(termQuery);
            //sourceBuilder.query(matchQuery);
            sourceBuilder.timeout(new TimeValue(60, TimeUnit.SECONDS));
            //执行搜索
            source = searchRequest.source(sourceBuilder);
            searchResponse = client.search(searchRequest, RequestOptions.DEFAULT);
            //解析结果


            for (SearchHit documentFields : searchResponse.getHits().getHits()) {
                list2.add(documentFields.getSourceAsMap());
            }
        };
        List<Map<String, Object>> list3 = new ArrayList<>();


        if(!list2.isEmpty()){
            Map<String, Object> map1 = new HashMap<String, Object>();
            map1.put("id",id);
            map1.put("question",question);
            map1.put("answer",answer);
            map1.put("all_answers",all_answers);
            map1.put("isBing",isBing);
            map1.put("context",context);
            list3.add(map1);
        }

        return list3;
    }

    public boolean writeQAContent() throws IOException {

        //write quesitons into ES
        String file_path="D:\\Download\\Springboot\\SearchEngine-v3\\all_groups_data.json";

        //调用JsonparseUtili解析数据
        List<QApair> questionList = new JsonParseUtil().parseJson(file_path);



        //把查询的数据放入 es 中
        BulkRequest request = new BulkRequest();
        request.timeout("2m");

        for (int i = 0; i < questionList.size(); i++) {
            request.add(
                    new IndexRequest("question_answer")
                            .source(JSON.toJSONString(questionList.get(i)), XContentType.JSON));

        }
        BulkResponse bulk = client.bulk(request, RequestOptions.DEFAULT);

        //write answers into ES

//        file_path="C:\\Users\\HP\\Desktop\\大三课程\\机器问答\\data\\answersnew.json";
//        List<Answer> answerList = new JsonParseUtil().parseAnJson(file_path);
//
//        //把查询的数据放入 es 中
//        request = new BulkRequest();
//        request.timeout("2m");
//
//        for (int i = 0; i < answerList.size(); i++) {
//            request.add(
//                    new IndexRequest("insurance_answer")
//                            .source(JSON.toJSONString(answerList.get(i)), XContentType.JSON));
//
//        }
//        bulk = client.bulk(request, RequestOptions.DEFAULT);

        return !bulk.hasFailures();
    }

}
