/**

The MIT License

Copyright (c) 2017 Rafael Muñoz-Salinas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include "fbow/vocabulary_creator.h"
#ifdef USE_OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads(){return 1;}
inline int omp_get_thread_num(){return 0;}
#endif
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
using namespace std;
namespace fbow{

std::vector<cv::Mat> readFeaturesFromFile(const std::string filename, std::string desc_name) {
    std::vector<cv::Mat> features;
    std::ifstream ifile(filename, std::ios::binary);
    if (!ifile.is_open()) {
        std::cerr << "could not open the input file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    char _desc_name[20];
    ifile.read(_desc_name, 20);
    desc_name = _desc_name;

    uint32_t size;
    ifile.read((char*) &size, sizeof(size));
    features.resize(size);
    for (size_t i = 0; i < size; i++) {
        uint32_t cols, rows, type;
        ifile.read((char*)&cols, sizeof(cols));
        ifile.read((char*)&rows, sizeof(rows));
        ifile.read((char*)&type, sizeof(type));
        features[i].create(rows, cols, type);
        ifile.read((char*)features[i].ptr<uchar>(0), features[i].total() * features[i].elemSize());
    }
    return features;
}

void VocabularyCreator::create(fbow::Vocabulary &Voc, const  cv::Mat  &features, const std::string &desc_name, Params params, bool use_idf)
{
    std::vector<cv::Mat> vfeatures(1);
    vfeatures[0]=features;
    create(Voc,vfeatures,desc_name,params, use_idf);
}

void VocabularyCreator::create(fbow::Vocabulary &Voc, const std::vector<cv::Mat> &features, const string &desc_name, Params params, bool use_idf){
    assert(features.size()>0);
    assert(features[0].cols>0);
    //select the funciton
    _params=params;
    _descCols=features[0].cols;
    _descType=features[0].type();
    _descNBytes=features[0].cols* features[0].elemSize();
    _params.nthreads=std::min(maxthreads,_params.nthreads);

    if(!(_descType==CV_8UC1|| _descType==CV_32FC1))
        throw std::runtime_error("Descriptors must be binary CV_8UC1 or float  CV_32FC1");
    if (_descType==CV_8UC1){
//        if (_descNBytes==32)dist_func=distance_hamming_32bytes;
//        else
            if(params.dist==0){
            dist_func=distance_hamming_generic;
            }else if(params.dist==1){
            dist_func=distance_jaccard_generic;
            }
    }
    else  dist_func=distance_float_generic;
    //create for later usage
    _features.create(features);

    //set all indices for the first level
    id_assigments.create(0,_features.size());
    auto root_assign=id_assigments[0];
    root_assign->resize (_features.size());
    for(size_t i=0;i<_features.size();i++) root_assign->at(i)=i;

    if(_params.nthreads>1){
        createLevel(0,0,false);
        //now, add threads

        for(auto &t:threadRunning)t=false;
        for(size_t i=0;i<_params.nthreads;i++)
            _Threads.push_back(std::thread(&VocabularyCreator::thread_consumer,this,i));
        int ntimes=0;
        while(ntimes++<10){
            for(auto &t:threadRunning) if (t){ ntimes=0;break;}
            std::this_thread::sleep_for(std::chrono::microseconds(600));
        }

        //add exit info
        for(size_t i=0;i<_Threads.size();i++) ParentDepth_ProcesQueue.push(std::make_pair(-1,-1));
        for(std::thread &th:_Threads) th.join();
    }
    else{
        createLevel(0,0,true);
    }
//    std::cout<<TheTree.size()<<std::endl;
//    for(auto &n:TheTree.getNodes())
//        std::cout<<n.first<<" ";std::cout<<std::endl;

    //now, transform the tree into a vocabulary
    convertIntoVoc(Voc,desc_name, features, use_idf);
}

void VocabularyCreator::thread_consumer(int idx){
    bool done=false;
    while(!done)
    {
        threadRunning[idx]=false;
        auto pair=ParentDepth_ProcesQueue.pop();//wait
        threadRunning[idx]=true;
//        std::cout<<"thread ("<<idx<<"):"<<pair.first<<std::endl;
        if (pair.first>=0)
            createLevel(pair.first,pair.second,false);
        else done=true;
    }
    threadRunning[idx]=false;
}


//ready to be threaded using producer consumer
void  VocabularyCreator::createLevel(  int parent, int curL,bool recursive){
    std::vector<cv::Mat> center_features;
    std::vector<vector_sptr > assigments_ref;
    assert(id_assigments.count(parent));
    const auto &findices=*id_assigments[parent];
    //trivial case, less features or equal than k (these are leaves)
    if ( findices.size()<=_params.k){
        for(auto fi:findices)
            center_features.push_back( _features[fi] );
    }
    else{
         //create the assigment vectors and reserve memory
        for(size_t i=0;i<_params.k;i++){
            id_assigments.create( parent*_params.k+1+i,findices.size()/_params.k);
            assigments_ref.push_back( id_assigments[ parent*_params.k+1+i]);
        }

        //initialize clusters
        auto centers=getInitialClusterCenters(findices );
        center_features.resize(centers.size());
        for(size_t i=0;i<centers.size();i++)
            center_features[i]=_features[centers[i]];
        //do k means evolution to move means
        size_t prev_hash=1,cur_hash=0;
        int niters=0;
        while(niters<_params.maxIters && cur_hash!=prev_hash ){
            std::swap(prev_hash,cur_hash);
            //do assigment
            assignToClusters(findices,center_features,assigments_ref /*,parent==0*/);
            //recompute centers again
            center_features=recomputeCenters(assigments_ref/*,parent==0*/);
            cur_hash=vhash(assigments_ref);
            niters++;
           };

        assignToClusters(findices,center_features,assigments_ref /*,parent==0*/);
        if (_params.verbose) std::cerr<<"Cluster created :"<<parent<<" "<<curL<<endl;

    }

    //add to the tree the set of nodes
    std::vector<Node> new_nodes;
    new_nodes.reserve(center_features.size());
    {
        for(size_t c=0;c<center_features.size();c++)
        new_nodes.push_back(Node(parent*_params.k+1+c,parent,center_features[c], findices.size()==center_features.size()?findices[c]:std::numeric_limits<uint32_t>::max()));
    }
    TheTree.add(new_nodes,parent);
    //we can now remove the assigments of the parent
    id_assigments.erase(parent);
  //  std::cout<<"parent "<<parent<<" done"<<std::endl;

    //should we go deeper?
    if ( ( (_params.L!=-1 && curL<(_params.L-1)) || _params.L==-1)  && assigments_ref.size()>0){
        assert(assigments_ref.size()==new_nodes.size());
        //go deeper again or add to queue
        if (recursive){//recursive mode(one thread only)
            for(size_t i=0;i<new_nodes.size();i++)
                createLevel( parent*_params.k+1+i,curL+1);
        }
        else{//parallel mode (multiple theads)
            //add as a item to be processed by a thread
            for(size_t i=0;i<new_nodes.size();i++)
                ParentDepth_ProcesQueue.push(std::make_pair(parent*_params.k+1+i,curL+1));
        }
    }
}

std::vector<uint32_t>  VocabularyCreator::getInitialClusterCenters(const std::vector<uint32_t> &findices  )
{

    //set distances to zero

    for(auto fi:findices) _features(fi).m_Dist=0;

    std::vector<uint32_t>   centers;
    centers.reserve(_params.k);

    // 1.Choose one center uniformly at random from among the data points.
    uint32_t ifeature = findices[rand()% findices.size()];
    // create first cluster
    centers.push_back(ifeature);
    do{
        // add the distance to the new cluster and select the farthest one
        auto last_center_feat=_features[centers.back()];
        std::pair<uint32_t,float> farthest(0,std::numeric_limits<float>::min());
        for(auto fi:findices){
            auto &feature=_features(fi);
            feature.m_Dist+=dist_func(last_center_feat, _features[fi]);
            if (feature.m_Dist>farthest.second)//found a farthest one
                farthest=std::make_pair(fi,feature.m_Dist);
        }
        ifeature=farthest.first;
        centers.push_back(ifeature);
    }while( centers.size() <_params.k);
    return centers;
}

std::size_t VocabularyCreator::vhash(const std::vector<std::vector<uint32_t> > & v_vec)  {
  std::size_t seed = 0;

  for(size_t i=0;i<v_vec.size();i++) seed+=v_vec[i].size()*(i+1);

  for(auto& vec : v_vec)
      for(auto& i : vec)
          seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);

  return seed;
}
std::size_t VocabularyCreator::vhash(const std::vector<vector_sptr> &v_vec)  {
  std::size_t seed = 0;

  for(size_t i=0;i<v_vec.size();i++) seed+=v_vec[i]->size()*(i+1);

  for(auto& vec : v_vec)
      for(auto& i : *vec)
          seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);

  return seed;
}



void VocabularyCreator::assignToClusters( const std::vector<uint32_t> &findices,  const std::vector<cv::Mat> &center_features,std::vector<vector_sptr> &assigments,bool omp){
    for(auto &a:assigments) a->clear();
    if(omp  ){
        std::vector<std::map<uint32_t,std::list<uint32_t> > >map_assigments_omp(omp_get_max_threads());
#pragma omp parallel for
        for(int i=0;i< int(findices.size());i++){
            auto tid=omp_get_thread_num();
            auto fi=findices[i];
            const auto &feature=_features[fi];
            std::pair<uint32_t,float> center_dist_min(0,dist_func(center_features[0],feature));
            for(size_t ci=1;ci<center_features.size();ci++){
                float dist=dist_func(center_features[ci],feature);
                if (dist< center_dist_min.second) center_dist_min=std::make_pair(ci,dist);
            }
            map_assigments_omp[tid][center_dist_min.first].push_back(fi);
//            assigments[center_dist_min.first]->push_back(fi);
        }
        //gather all assignments in output
        for(const auto &mas_tid:map_assigments_omp){
            for(const auto &c_assl:mas_tid){
                for(const auto &id:c_assl.second)
                    assigments[c_assl.first]->push_back(id);
            }
        }
    }
    else{

        for(auto fi: findices){
            const auto &feature=_features[fi];
            std::pair<uint32_t,float> center_dist_min(0,dist_func(center_features[0],feature));
            for(size_t ci=1;ci<center_features.size();ci++){
                float dist=dist_func(center_features[ci],feature);
                if (dist< center_dist_min.second) center_dist_min=std::make_pair(ci,dist);
            }
            assigments[center_dist_min.first]->push_back(fi);
        }
    }
    //check
    //    for(int i=0;i<assigments.size();i++)
    //        for(int j=0;j<assigments.size();j++){
    //            if(i!=j){
    //                for(auto c:*assigments[i])
    //                    assert(std::find(assigments[j]->begin(),assigments[j]->end(),c)==assigments[j]->end());
    //            }
    //        }
}
/**
 * @brief VocabularyCreator::recomputeCenters
 * @param findices
 * @param features
 * @param assigments
 * @return
 */


std::vector<cv::Mat> VocabularyCreator::recomputeCenters(  const std::vector<vector_sptr> &assigments,bool omp){
std::vector<cv::Mat> centers;
if (omp){
    centers.resize(assigments.size());
#pragma omp parallel for
    for(int i=0;i<int(assigments.size());i++){
        if (_descType==CV_8UC1)   centers[i]=meanValue_binary(*assigments[i]);
        else centers[i]=meanValue_float(*assigments[i]) ;
    }
}
else{
    centers.reserve(assigments.size());
    for(const auto &ass:assigments){
        if (_descType==CV_8UC1)   centers.push_back(meanValue_binary(*ass) );
        else centers.push_back(meanValue_float(*ass) );
    }
}
    return centers;
}
cv::Mat VocabularyCreator::meanValue_binary( const std::vector<uint32_t>  &indices)
{

    //determine number of bytes of the binary descriptor
     std::vector<int> sum( _descNBytes * 8, 0);

    for(auto i:indices)
    {
        const unsigned char *p = _features[i].ptr<unsigned char>();
         for(int j = 0; j < _descCols; ++j, ++p)
        {
            if(*p & 128) ++sum[ j*8     ];
            if(*p &  64) ++sum[ j*8 + 1 ];
            if(*p &  32) ++sum[ j*8 + 2 ];
            if(*p &  16) ++sum[ j*8 + 3 ];
            if(*p &   8) ++sum[ j*8 + 4 ];
            if(*p &   4) ++sum[ j*8 + 5 ];
            if(*p &   2) ++sum[ j*8 + 6 ];
            if(*p &   1) ++sum[ j*8 + 7 ];
        }
    }

    cv::Mat mean = cv::Mat::zeros(1, _descNBytes, CV_8U);
    unsigned char *p = mean.ptr<unsigned char>();

    const int N2 = (int)indices.size() / 2 + indices.size() % 2;
    for(size_t i = 0; i < sum.size(); ++i)
    {
        // set bit
        if(sum[i] >= N2) *p |= 1 << (7 - (i % 8));
        if(i % 8 == 7) ++p;
    }
return mean;
}

cv::Mat VocabularyCreator::meanValue_float( const std::vector<uint32_t>  &indices){
    cv::Mat mean(1,_descCols,_descType);
    mean.setTo(cv::Scalar::all(0));
    for(auto i:indices) mean +=  _features[i] ;
    mean*= 1./double( indices.size());

    return mean;
}


void VocabularyCreator::convertIntoVoc(Vocabulary &Voc,  std::string  desc_name, const std::vector<cv::Mat>& features, bool use_idf){

    //look for leafs and store
    //now, create the blocks

    uint32_t nLeafNodes=0;
    uint32_t nonLeafNodes=0;
    std::map<uint32_t,uint32_t> nodeid_blockid;
    for(auto &node:TheTree.getNodes()){
            if(node.second.isLeaf())
            {
                //assing an id if not set
                if ( node.second.feat_idx==std::numeric_limits<uint32_t>::max()) node.second.feat_idx=nLeafNodes;
                nLeafNodes++;
            }
            else nodeid_blockid.insert(std::make_pair(node.first,nonLeafNodes++));

    }
    //determine the basic elements
    Voc.clear();
    int aligment=8;
    if (_descType==CV_32F) aligment=32;
    Voc.setParams(aligment,_params.k,_descType,_descNBytes,nonLeafNodes,desc_name);

    //lets start
    for(auto &node:TheTree.getNodes()){
        if (!node.second.isLeaf()){
            auto binfo=Voc.getBlock(nodeid_blockid[node.first]);
            binfo.setN(node.second.children.size());
            binfo.setParentId(node.first);
            bool areAllChildrenLeaf=true;
            for(size_t c=0;c< node.second.children.size();c++){
                Node &child=TheTree.getNodes()[node.second.children[c]];
                binfo.setFeature(c,child.feature);
                //go to the end and set info
                if (child.isLeaf())   binfo.getBlockNodeInfo(c)->setLeaf(child.feat_idx,child.weight);
                else {
                    binfo.getBlockNodeInfo(c)->setNonLeaf(nodeid_blockid[child.id]);
                    areAllChildrenLeaf=false;
                }
            }
            binfo.setLeaf(areAllChildrenLeaf);
        }
    }
    if(use_idf){
        std::cout << "Finished creating Vocabulary. Starting IDF weigth calculation." << std::endl;
        // Voc is created but weights are only based on crispness until here. 
        // to get the needed IDF involved, we need to adjust the weights based on number of occurences of each word in an image test set
        std::map<int, int> words_number_of_occurances;
        int number_of_images = features.size();

        // transform IDF transforms the image and stores the amount of features in the image and features per word
        //  we perform this for a set of X images and use the stored values to determine the two following parameters:
        // For any "visual word", the Document Frequency (DF) is the number of images containing this "visual word" divided by the total number of images. The IDF is the inverse of this value.

        for(auto f : features){
            std::map<int, bool> feature_occured_map=Voc.transformIDF(f);
            for (auto i : feature_occured_map){
                if (!words_number_of_occurances.count(i.first)) {
                // not found -> word has not occured yet and has to be added to the map
                words_number_of_occurances.insert({i.first, 1});
                } else {
                // found -> word already in map: we increment the count
                words_number_of_occurances[i.first]+=1;
                }
            } 
        }

        //-> go into each leaf and change the weight to IDF

        float idf;
        for(auto &node:TheTree.getNodes()){
            if (!node.second.isLeaf()){
                auto binfo=Voc.getBlock(nodeid_blockid[node.first]);
                binfo.setN(node.second.children.size());
                binfo.setParentId(node.first);
                bool areAllChildrenLeaf=true;
                for(size_t c=0;c< node.second.children.size();c++){
                    Node &child=TheTree.getNodes()[node.second.children[c]];
                    binfo.setFeature(c,child.feature);
                    //go to the end and set info
                    if (child.isLeaf()){
                        // calculating IDF of the word
                        idf = log((float)number_of_images/(float)words_number_of_occurances[child.feat_idx]);
                        if(!words_number_of_occurances.count(child.feat_idx)){
                            std::cout << "Word not in map" << std::endl;
                            idf = log((float)number_of_images);
                        }
                        binfo.getBlockNodeInfo(c)->setLeaf(child.feat_idx, idf);
                    }
                    else {
                        binfo.getBlockNodeInfo(c)->setNonLeaf(nodeid_blockid[child.id]);
                        areAllChildrenLeaf=false;
                    }
                }
                binfo.setLeaf(areAllChildrenLeaf);
            }
        }
        std::cout << "Images: " << number_of_images << std::endl;
    }
}

float VocabularyCreator::distance_hamming_generic(const cv::Mat &a, const cv::Mat &b){
    uint64_t ret=0;
    const uchar *pa = a.ptr<uchar>(); // a & b are actually CV_8U
    const uchar *pb = b.ptr<uchar>();
    for(int i=0;i<a.cols;i++,pa++,pb++){
        uchar v=(*pa)^(*pb);
#ifdef __GNUG__
        ret+=__builtin_popcount(v);//only in g++
#else
        ret+=(v& (1))!=0;
        ret+=(v& (2))!=0;
        ret+=(v& (4))!=0;
        ret+=(v& (8))!=0;
        ret+=(v& (16))!=0;
        ret+=(v& (32))!=0;
        ret+=(v& (64))!=0;
        ret+=(v& (128))!=0;
#endif
    }
    return ret;
}

float VocabularyCreator::distance_jaccard_generic(const cv::Mat &a, const cv::Mat &b){
    float ret=0;
    uint64_t un=0;
    uint64_t inter=0;
    const uchar *pa = a.ptr<uchar>(); // a & b are actually CV_8U
    const uchar *pb = b.ptr<uchar>();
    for(int i=0;i<a.cols;i++,pa++,pb++){
        uchar u=(*pa)|(*pb);
        uchar in=(*pa)&(*pb);
#ifdef __GNUG__
        un+=__builtin_popcount(u);//only in g++
        inter+=__builtin_popcount(in);
#else
        ret+=(v& (1))!=0;
        ret+=(v& (2))!=0;
        ret+=(v& (4))!=0;
        ret+=(v& (8))!=0;
        ret+=(v& (16))!=0;
        ret+=(v& (32))!=0;
        ret+=(v& (64))!=0;
        ret+=(v& (128))!=0;
#endif
    }
    ret=1.0-((float)inter/(float)un);
    return ret;
}

//for orb
float VocabularyCreator::distance_hamming_32bytes(const cv::Mat &a, const cv::Mat &b){
    std::cout << "HAMMING" << std::endl;
    const uint64_t *pa = a.ptr<uint64_t>(); // a & b are actually CV_8U
    const uint64_t *pb = b.ptr<uint64_t>();
    return uint64_popcnt(pa[0]^pb[0])+ uint64_popcnt(pa[1]^pb[1])+  uint64_popcnt(pa[2]^pb[2])+uint64_popcnt(pa[3]^pb[3]);
}
float VocabularyCreator::distance_float_generic(const cv::Mat &a, const cv::Mat &b){
    std::cout << "FLOAT" << std::endl;
    double sqd = 0.;
    const float *a_ptr=a.ptr<float>(0);
    const float *b_ptr=b.ptr<float>(0);
    for(int i = 0; i < a.cols; i ++) sqd += (a_ptr[i  ] - b_ptr[i  ])*(a_ptr[i  ] - b_ptr[i  ]);
    std::cout << sqd << std::endl;
    return sqd;
}


}

