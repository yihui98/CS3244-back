(this.webpackJsonpfront=this.webpackJsonpfront||[]).push([[0],{166:function(e,t,a){"use strict";a.r(t);var n=a(0),r=a(28),i=a.n(r),s=a(44),o=a.n(s),c=a(64),l=a(13),d=a(84),h=a(4),m=a(243),p=a(234),b=a(250),j=a(244),u=a(246),g=a(251),x=a(238),f=a(104),w=a.n(f),y=a(247),O=a(245),v=a(105),k=a.n(v),W=a(106),B=a.n(W),S=a(6),F=a(239),L=a(227),N=a(242),C=a(249),D=a(248),T=a(107),E=a.n(T),M=a(237),R=a(103),I=a.n(R),A=function(){var e=Object(c.a)(o.a.mark((function e(t){var a;return o.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.prev=0,e.next=3,I.a.get("".concat("/model","/").concat(t));case 3:return a=e.sent,e.abrupt("return",a.data);case 7:return e.prev=7,e.t0=e.catch(0),e.abrupt("return",e.t0);case 10:case"end":return e.stop()}}),e,null,[[0,7]])})));return function(t){return e.apply(this,arguments)}}(),z={getSentiment:A},H=a(38),P=a(252),G=a(108),J=a.n(G),U=a(223),_=a(224),q=a.p+"static/media/workflow.9e301a6a.png",K=a.p+"static/media/eda.edb0cdcb.png",Q=a.p+"static/media/preprocessingSummary.9febbf4f.png",V=a.p+"static/media/newFeatures.b41d6276.png",X=a.p+"static/media/heatmap.453d6504.png",Y=a.p+"static/media/featuresSummary.b23cb91c.png",Z=a.p+"static/media/linear.179484da.png",$=a.p+"static/media/ensemble.7037f95b.png",ee=a.p+"static/media/deep.95e825a5.png",te=a.p+"static/media/confusion.c0f913a1.png",ae=a.p+"static/media/words.3a69c59d.png",ne=a(2);function re(e){var t=e.children,a=e.window,n=Object(U.a)({target:a?a():void 0});return Object(ne.jsx)(_.a,{appear:!1,direction:"down",in:!n,children:t})}var ie=Object(F.a)((function(e){return{root:{flexGrow:1,color:"linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)"},menuButton:{marginRight:e.spacing(2)},title:Object(h.a)({flexGrow:1,display:"none"},e.breakpoints.up("sm"),{display:"block"}),search:Object(h.a)({position:"relative",borderRadius:e.shape.borderRadius,marginLeft:0,width:"100%"},e.breakpoints.up("sm"),{marginLeft:e.spacing(1),width:"auto"}),searchIcon:{padding:e.spacing(0,2),height:"100%",position:"absolute",pointerEvents:"none",display:"flex",alignItems:"center",justifyContent:"center"},inputRoot:{color:"linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)"},drawer:{width:200,flexShrink:0},drawerHeader:Object(d.a)(Object(d.a)({display:"flex",alignItems:"center",padding:e.spacing(0,1)},e.mixins.toolbar),{},{justifyContent:"flex-start"}),drawerPaper:{width:200},inputInput:Object(h.a)({padding:e.spacing(1,1,1,0),paddingLeft:"calc(1em + ".concat(e.spacing(4),"px)"),transition:e.transitions.create("width"),width:"100%"},e.breakpoints.up("sm"),{width:"12ch","&:focus":{width:"20ch"}})}}));var se=function(){var e=Object(n.useState)(""),t=Object(l.a)(e,2),a=t[0],r=t[1],i=Object(n.useState)("Waiting for input"),s=Object(l.a)(i,2),d=s[0],h=s[1],f=Object(n.useState)(!1),v=Object(l.a)(f,2),W=v[0],F=v[1],T=ie(),R=Object(L.a)(),I=function(){var e=Object(c.a)(o.a.mark((function e(){var t;return o.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:if(""!==a){e.next=4;break}h("Waiting for input"),e.next=8;break;case 4:return e.next=6,z.getSentiment(a);case 6:t=e.sent,h(t.score);case 8:case"end":return e.stop()}}),e)})));return function(){return e.apply(this,arguments)}}();return Object(ne.jsxs)("div",{id:"top",children:[Object(ne.jsx)(re,{children:Object(ne.jsx)(m.a,{position:"sticky",style:{background:"linear-gradient(45deg, #249cdb 30%, #2dd27e 90%)"},children:Object(ne.jsxs)(j.a,{children:[Object(ne.jsx)(O.a,{color:"inherit","aria-label":"open drawer",onClick:function(){F(!0)},edge:"start",className:Object(S.a)(T.menuButton,W&&T.hide),children:Object(ne.jsx)(w.a,{})}),Object(ne.jsx)(u.a,{className:T.title,variant:"h6",noWrap:!0,children:"Team 34"})]})})}),Object(ne.jsxs)(y.a,{className:T.drawer,variant:"persistent",anchor:"left",open:W,classes:{paper:T.drawerPaper},children:[Object(ne.jsx)("div",{className:T.drawerHeader,children:Object(ne.jsx)(O.a,{onClick:function(){F(!1)},children:"ltr"===R.direction?Object(ne.jsx)(k.a,{}):Object(ne.jsx)(B.a,{})})}),Object(ne.jsxs)(D.a,{children:[Object(ne.jsx)(H.Link,{to:"top",spy:!0,smooth:!0,style:{textDecoration:"none",color:"black"},children:Object(ne.jsx)(N.a,{button:!0,children:Object(ne.jsx)(C.a,{primary:"Home"})})}),Object(ne.jsx)(H.Link,{to:"motivation",spy:!0,smooth:!0,style:{textDecoration:"none",color:"black"},children:Object(ne.jsx)(N.a,{button:!0,children:Object(ne.jsx)(C.a,{primary:"Motivation"})})}),Object(ne.jsx)(H.Link,{to:"methods",spy:!0,smooth:!0,style:{textDecoration:"none",color:"black"},children:Object(ne.jsx)(N.a,{button:!0,children:Object(ne.jsx)(C.a,{primary:"Methods"})})}),Object(ne.jsx)(H.Link,{to:"preprocessing",spy:!0,smooth:!0,style:{textDecoration:"none",color:"black"},children:Object(ne.jsx)(N.a,{button:!0,children:Object(ne.jsx)(C.a,{primary:"Preprocessing"})})}),Object(ne.jsx)(H.Link,{to:"models",spy:!0,smooth:!0,style:{textDecoration:"none",color:"black"},children:Object(ne.jsx)(N.a,{button:!0,children:Object(ne.jsx)(C.a,{primary:"Models"})})}),Object(ne.jsx)(H.Link,{to:"conclusion",spy:!0,smooth:!0,style:{textDecoration:"none",color:"black"},children:Object(ne.jsx)(N.a,{button:!0,children:Object(ne.jsx)(C.a,{primary:"Conclusion"})})})]})]}),Object(ne.jsx)(b.a,{sx:{bgcolor:"background.paper",pt:8,pb:6},children:Object(ne.jsxs)(g.a,{maxWidth:"sm",children:[Object(ne.jsx)(u.a,{component:"h1",variant:"h2",align:"center",color:"text.primary",gutterBottom:!0,children:"CS3244 Sarcasm Model"}),Object(ne.jsxs)(u.a,{variant:"h5",align:"center",color:"text.secondary",paragraph:!0,children:["This model was created by training the data on the ",Object(ne.jsx)("a",{href:"https://www.kaggle.com/danofer/sarcasm",style:{"text-decoration":"none",color:"inherit"},children:"Reddit sarcasm dataset "})]}),Object(ne.jsxs)(p.a,{sx:{pt:4},direction:"row",spacing:2,justifyContent:"center",children:[Object(ne.jsx)(x.a,{fullWidth:!0,sx:{m:1},id:"outlined-basic",label:"Try it now! Input your text here",variant:"outlined",value:a,onChange:function(e){console.log(e.target.value),r(e.target.value)},multiline:!0}),Object(ne.jsx)(M.a,{title:"Generate Sentiment",children:Object(ne.jsx)(O.a,{"aria-label":"add",onClick:I,children:Object(ne.jsx)(E.a,{})})})]}),Object(ne.jsxs)(p.a,{sx:{pt:4},direction:"row",spacing:2,justifyContent:"center",children:[Object(ne.jsx)("div",{children:" Score : "}),Object(ne.jsxs)("div",{children:[" ",d," "]})]})]})}),Object(ne.jsxs)(b.a,{children:[Object(ne.jsxs)(g.a,{sx:{py:8},maxWidth:"md",id:"motivation",children:[Object(ne.jsx)(u.a,{component:"h1",variant:"h2",align:"center",color:"text.primary",gutterBottom:!0,children:" Motivation"}),Object(ne.jsx)("div",{children:" Misunderstandings and arguments can result from readers\u2019 misinterpretation of a sarcastic remark as a serious one. As such, we aim to create a one-size-fits-all sarcasm detection model to enables writers, on different platforms such as Reddit or traditional news media, to comment freely without the fear of offending someone else."})]}),Object(ne.jsxs)(g.a,{sx:{py:2},maxWidth:"md",id:"methods",children:[Object(ne.jsx)(u.a,{component:"h1",variant:"h2",align:"center",color:"text.primary",gutterBottom:!0,children:" Our Methods"}),Object(ne.jsxs)("div",{style:{"white-space":"pre-wrap"},children:["Our general approach to tackling this problem is by splitting our workload into 2 parts; 2 of our members will be in charge of data processing while the other 3 will be working on the models. ","\n"," ","\n","Data processing contains of several components such as data cleaning, data preprocessing, feature engineering and finally feature selection. ","\n"," ","\n","While data modeling entails training and testing a suite of models, hyperparameter tuning, evaluation and analysis of the model's performance. The high-level workflow is shown below. ","\n"," ","\n"]}),Object(ne.jsx)("img",{src:q,alt:"Logo",maxWidth:"md",style:{width:"100%",height:"100%"}})]}),Object(ne.jsxs)(g.a,{sx:{py:2},maxWidth:"md",id:"preprocessing",children:[Object(ne.jsx)(u.a,{component:"h1",variant:"h2",align:"center",color:"text.primary",gutterBottom:!0,children:" Preprocessing"}),Object(ne.jsxs)("div",{style:{"white-space":"pre-wrap"},children:["Firstly, we begin by examining our dataset. By performing an exploratory data analysis, we are able to identify the type of data that we are working with. A summary of our dataset is shown below ","\n"," ","\n",Object(ne.jsx)("img",{src:K,alt:"eda",maxWidth:"md",style:{width:"100%",height:"100%"}})," ","\n"," ","\n","We can see that after filtering, the class distribution remains balanced. We now perform a wide range of data preprocessing methods such as Word Normalization, N-grams and TF-IDF. A summary of our results can be seen below ","\n"," ","\n",Object(ne.jsx)("img",{src:Q,alt:"preprocessingSummary",maxWidth:"md",style:{width:"100%",height:"100%"}})," ","\n"," ","\n","Finally, we proceed on with feature engineering and feature selection. A summary of our top distinguishing new features can be seen below.",Object(ne.jsx)("img",{src:V,alt:"newFeatures",maxWidth:"md",style:{width:"100%",height:"100%"}})," ","\n"," ","\n","We are able to see that no single feature has a large distinct horizontal non-overlap and if we include every new feature, it will add too much noise to our data. As such, we decided to limit features that have the least overlap in histogram plots between the two classes.",Object(ne.jsx)("img",{src:X,alt:"heatmap",maxWidth:"md",style:{width:"100%",height:"100%"}})," ","\n"," ","\n","A summary of our results can be seen below. ","\n"," ","\n",Object(ne.jsx)("img",{src:Y,alt:"featuresSummary",maxWidth:"md",style:{width:"100%",height:"100%"}})," ","\n"," ","\n","As per our initial hypothesis of lack of disjointness of classes in new features, adding any of the new feature tend to worsen performance. Some features like 'avg_wordlength' will heavily skew prediction of the test set towards one class. Thus, we decided to stick to our initial bag-of-words, 1-2 grams approach for better generalization of new data."]})]}),Object(ne.jsxs)(g.a,{sx:{py:2},maxWidth:"md",id:"models",children:[Object(ne.jsx)(u.a,{component:"h1",variant:"h2",align:"center",color:"text.primary",gutterBottom:!0,children:" Models "}),Object(ne.jsxs)("div",{style:{"white-space":"pre-wrap"},children:["Using the preprocessed data, we perform training on the dataset by using a wide range of models from linear to deeplearning. This includes linear model such as Logistic Regression, ensemble methods such as Random Forest classification and deep learning methods such as Recurrent Neural Networks. ","\n"," ","\n","Our results are as follows: ","\n"," ","\n",Object(ne.jsx)(u.a,{component:"h4",variant:"h4",color:"text.primary",gutterBottom:!0,children:" Linear / Non-linear models: "}),Object(ne.jsx)("img",{src:Z,alt:"linear",maxWidth:"md",style:{width:"100%",height:"100%"}})," ","\n"," ","\n",Object(ne.jsx)(u.a,{component:"h4",variant:"h4",color:"text.primary",gutterBottom:!0,children:" Ensemble Methods: "}),"  ","\n"," ","\n",Object(ne.jsx)("img",{src:$,alt:"ensemble",maxWidth:"md",style:{width:"100%",height:"100%"}})," ","\n"," ","\n",Object(ne.jsx)(u.a,{component:"h4",variant:"h4",color:"text.primary",gutterBottom:!0,children:" Deep Learning: "})," ","\n"," ","\n",Object(ne.jsx)("img",{src:ee,alt:"deep",maxWidth:"md",style:{width:"100%",height:"100%"}})," ","\n"," ","\n"]})]}),Object(ne.jsxs)(g.a,{sx:{py:2},maxWidth:"md",id:"conclusion",children:[Object(ne.jsx)(u.a,{component:"h1",variant:"h2",align:"center",color:"text.primary",gutterBottom:!0,children:" Conclusion "}),Object(ne.jsxs)("div",{style:{"white-space":"pre-wrap"},children:["Can our best models generalise well to other sarcastic contents such as News Headlines? ","\n"," ","\n","Sadly no, a 83% accuracy ExtraTrees Ensemble Model can only predict 51.7% of the News headlines correctly. This is only slightly better than making a randomized choice. ","\n"," ","\n",Object(ne.jsx)("img",{src:te,maxWidth:"md",alt:"confusion",style:{width:"50%",height:"100%"}})," ","\n"," ","\n","One reason for this could be because the important words found in News headlines do not exist in the TF-IDF features. If we take a look at the words found in News headlines vs Reddit comments, we can see that News headlines are generally more formal and made up of actual English words. ","\n"," ","\n",Object(ne.jsx)("img",{src:ae,maxWidth:"md",alt:"words",style:{width:"100%",height:"100%"}})," ","\n"," ","\n","To improve on the accuracy, in the future we will train our models on both formal and informal datasets so that our models are able to learn the context of the data and hopefully produce better results."]})]}),Object(ne.jsx)(H.Link,{to:"top",spy:!0,smooth:!0,style:{textDecoration:"none",color:"black"},children:Object(ne.jsx)(P.a,{color:"primary",size:"medium","aria-label":"scroll back to top",style:{position:"sticky",bottom:"8%",left:"92%"},children:Object(ne.jsx)(J.a,{})})})]})]})};i.a.render(Object(ne.jsx)(se,{}),document.getElementById("root"))}},[[166,1,2]]]);
//# sourceMappingURL=main.ab7b259e.chunk.js.map