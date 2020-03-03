#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "flashgg/DataFormats/interface/Jet.h"
#include "flashgg/DataFormats/interface/DiPhotonCandidate.h"
#include "flashgg/DataFormats/interface/THQLeptonicTag.h"
//#include "flashgg/DataFormats/interface/THQLeptonicMVAResult.h"
#include "flashgg/DataFormats/interface/Electron.h"
#include "flashgg/DataFormats/interface/Muon.h"
#include "flashgg/DataFormats/interface/Photon.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
//#include "DataFormats/PatCandidates/interface/MET.h"

#include "flashgg/DataFormats/interface/Met.h"

#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "flashgg/Taggers/interface/LeptonSelection2018.h"

#include "DataFormats/Math/interface/deltaR.h"

//#include "flashgg/DataFormats/interface/TagTruthBase.h"
#include "flashgg/DataFormats/interface/THQLeptonicTagTruth.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "flashgg/Taggers/interface/SemiLepTopQuark.h"
#include "PhysicsTools/CandUtils/interface/EventShapeVariables.h"
#include "flashgg/Taggers/interface/FoxWolfram.hpp"

#include "flashgg/DataFormats/interface/PDFWeightObject.h"

#include <vector>
#include <algorithm>
#include <string>
#include <utility>
#include "TLorentzVector.h"
#include "TMath.h"
#include "TMVA/Reader.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TCanvas.h"
#include <map>

#include "flashgg/Taggers/interface/TTH_DNN_Helper.h"
// https://github.com/cms-analysis/flashgg/commit/f327ca16c29b4ced8eaf8c309cb9218fac265963 (fixing the tth taggers)
using namespace std;
using namespace edm;

// Declare function
int neutrino_reco(TLorentzVector electronLorentz, TLorentzVector missingEnergyLorentz,
                  TLorentzVector & neutrino_solution_1, TLorentzVector & neutrino_solution_2);

// W-boson mass
const float mass_w = 80.399;

namespace flashgg {
  class CTCVWeightedVariable {
  public:
    CTCVWeightedVariable( string name , string title , int nBins , double min , double max ) {
      Name = name;
      edm::Service<TFileService> fs;
      Directory = fs->mkdir( name ) ; 
      for (uint i = 0 ; i < 70 ; i++) {
        Histos.push_back( Directory.make< TH1D >( ("ctcv_"+to_string(i)).c_str() , (title + "," + to_string(i)).c_str() , nBins , min, max ) );
      }
    };

    void Fill( double value , std::vector<double> weights) {
      Histos[0]->Fill( value );
      for( uint i = 0 ; i < weights.size() ; i++) {
        Histos[i+1]->Fill( value , weights[i] );
      }
    };

    void Write() {
      Directory.make< TCanvas >( ("Canvas_"+Name).c_str() );
      for( auto h : Histos ) {
    	h->DrawNormalized();
      }
    }

    TFileDirectory Directory;
    vector< TH1* > Histos ;
    string Name;
    
  };

  class THQLeptonicTagProducer : public EDProducer {

    public:
      typedef math::XYZPoint Point;
      map< string , CTCVWeightedVariable* > CTCVWeightedVariables;
    
      THQLeptonicTagProducer( const ParameterSet & );

    private:
      std::string processId_;
      // edm::EDGetTokenT< LHEEventProduct > token_lhe;
      void produce( Event &, const EventSetup & ) override;
      virtual void beginJob() override {
        if(processId_.find("thq") != std::string::npos or processId_.find("thw") != std::string::npos){
        }
      };

      virtual void endJob() override {
      };

      std::vector<edm::EDGetTokenT<View<flashgg::Jet> > > tokenJets_;
      EDGetTokenT<View<DiPhotonCandidate> > diPhotonToken_;
      std::vector<edm::InputTag> inputTagJets_;
      EDGetTokenT<View<Electron> > electronToken_;
      EDGetTokenT<View<flashgg::Muon> > muonToken_;
      EDGetTokenT<View<DiPhotonMVAResult> > mvaResultToken_;
      EDGetTokenT<View<Photon> > photonToken_;
      EDGetTokenT<View<reco::Vertex> > vertexToken_;
      EDGetTokenT<View<flashgg::Met> > METToken_;
      EDGetTokenT<View<reco::GenParticle> > genParticleToken_;
      EDGetTokenT<View<reco::GenJet> > genJetToken_;
      edm::EDGetTokenT<vector<flashgg::PDFWeightObject> > weightToken_;
      EDGetTokenT<double> rhoTag_;
      string systLabel_;

      typedef std::vector<edm::Handle<edm::View<flashgg::Jet> > > JetCollectionVector;

      //Thresholds
      double leptonPtThreshold_;
      double leptonEtaThreshold_;
      vector<double> electronEtaThresholds_;
      double leadPhoOverMassThreshold_;
      double subleadPhoOverMassThreshold_;
      double MVAThreshold_;
      double deltaRLepPhoThreshold_;
      double deltaRJetLepThreshold_;

      double deltaRJetLeadPhoThreshold_;
      double deltaRJetSubLeadPhoThreshold_;

      double jetsNumberThreshold_;
      double bjetsNumberThreshold_;
      double jetPtThreshold_;
      double jetEtaThreshold_;

      vector<double> bDiscriminator_;
      string bTag_;
      double muPFIsoSumRelThreshold_;
      double PhoMVAThreshold_;
      double DeltaRTrkElec_;

      double deltaRPhoElectronThreshold_;
      double Zmass_;
      double deltaMassElectronZThreshold_;

      double MuonEtaCut_;
      double MuonPtCut_;
      double MuonIsoCut_;
      double MuonPhotonDrCut_;

      int    MinNLep_;
      double LeptonsZMassCut_;

      bool hasGoodElec = false;  bool hasVetoElec = false;
      bool hasGoodMuons = false;

      FileInPath tthVstHDNNfile_;
      std::vector<double> tthVstHDNN_global_mean_;
      std::vector<double> tthVstHDNN_global_stddev_;
      std::vector<double> tthVstHDNN_object_mean_;
      std::vector<double> tthVstHDNN_object_stddev_; 

      unique_ptr<TMVA::Reader> thqLeptonicMva_;
      FileInPath thqLeptonicMVAweightfile_;
      string  MVAMethod_;
      Float_t thqLeptonicMvaResult_value_, topMass, topMassTransverse;

      std::vector< TLorentzVector > particles_LorentzVector; 
      std::vector< math::RhoEtaPhiVector > particles_RhoEtaPhiVector;
        
      TLorentzVector metL, metW_check, bL,fwdJL, G1, G2, bjetL, tH_recoL;  //temp solution: make met, bjet & jprime global TLorentzVectors

      struct GreaterByPt {
        public:
          bool operator()( edm::Ptr<flashgg::Jet> lh, edm::Ptr<flashgg::Jet> rh ) const {
	    return lh->pt() > rh->pt();
          };
      };
        
      struct GreaterByEta {
        public:
          bool operator()( edm::Ptr<flashgg::Jet> lh, edm::Ptr<flashgg::Jet> rh ) const {
	    return fabs(lh->eta()) > fabs(rh->eta());
          };
      };
      
      struct GreaterByBTagging
      {
      public:
        GreaterByBTagging(std::string urName, std::string urName1):
            urName( urName ), urName1(urName1)
        {
        }

        bool operator()( edm::Ptr<flashgg::Jet> lh, edm::Ptr<flashgg::Jet> rh ) const
        {
            return (lh->bDiscriminator(urName.data()) + lh->bDiscriminator(urName1.data())) > (rh->bDiscriminator(urName.data()) + rh->bDiscriminator(urName1.data())) ;
        };
      private:
        const std::string urName, urName1;
        //	const std::string urName1;
      };

      int LeptonType;
      std::vector<edm::Ptr<flashgg::Jet> > SelJetVect;
      std::vector<edm::Ptr<flashgg::Jet> > SelJetVect_EtaSorted;
      std::vector<edm::Ptr<flashgg::Jet> > SelJetVect_PtSorted;
      std::vector<edm::Ptr<flashgg::Jet> > SelJetVect_BSorted;
      std::vector<edm::Ptr<flashgg::Jet> > MediumBJetVect, MediumBJetVect_PtSorted;
      std::vector<edm::Ptr<flashgg::Jet> > LooseBJetVect, LooseBJetVect_PtSorted ;
      std::vector<edm::Ptr<flashgg::Jet> > TightBJetVect, TightBJetVect_PtSorted;


      edm::Ptr<flashgg::Jet> fwdJet;
      edm::Ptr<flashgg::Jet> bJet  ;
      
      void topReco( std::vector<edm::Ptr<flashgg::Jet> >* bjets ) {
          
          topMass = -100.;
          topMassTransverse = -100.;
          thqLeptonicMvaResult_value_ = -100.;

          if ( bjets->size() < 1 || SelJetVect.size() < 2 || LeptonType == 0) {
              return;
          }
        
          fwdJet = SelJetVect_EtaSorted[0];
          bJet = bjets->at(0);
        
          if( fwdJet == bJet )
	          fwdJet = SelJetVect_EtaSorted[1] ;

          bL.SetPtEtaPhiE( bJet->pt(), bJet->eta(), bJet->phi(), bJet->energy());
          fwdJL.SetPtEtaPhiE( fwdJet->pt(),fwdJet->eta(), fwdJet->phi(), fwdJet->energy());

          flashgg::SemiLepTopQuark singletop(bL, metL, lepton.LorentzVector(), fwdJL,fwdJL);
          n_jets = SelJetVect.size();
          metL = singletop.getMET() ;
          jprime_eta  = fabs( fwdJL.Eta() );
          met_pt = metL.Pt();
          metW_check = singletop.neutrino_W () ;
          topMass = singletop.top().M() ;
          topMassTransverse = singletop.top().Mt() ;
          bjet1_pt = bJet->pt();
                
          fwdjet1_fabseta = fabs(SelJetVect_EtaSorted[0]->eta());
      
          // to may coincidence with trining names (FIX!)
          bjet_mult = num_bjets;
          top_mass_transverse = topMassTransverse;   
          bjet1_discr = bDiscriminatorValue;
          dR_lepton_fwdj = dR_lepton_fwdjet;
          n_central_jets = number_of_central_jets;
          dR_leadpho_fwdj = dR_leadpho_fwdjet;
          dR_subleadpho_fwdj = dR_subleadpho_fwdjet;
          deta_lepton_fwdj = dEta_lepton_fwdjet;
          recoMET_pt = met_pt;     
          n_forward_jets = number_of_forward_jets;
          dipho_dR = dR_leadPho_subleadPho;
          dR_dipho_fwdj = dR_dipho_fwdjet;

          if (MVAMethod_ != "") 
              thqLeptonicMvaResult_value_ = thqLeptonicMva_->EvaluateMVA( MVAMethod_.c_str() );
      };

      float dnn_score_0_;
      float ttH_vs_tH_dnn_score;

      TTH_DNN_Helper* dnn;
      TTH_DNN_Helper* dnn_ttH_vs_tH;

      // MVA INPUTS
      float jprime_eta;

      Float_t met_pt = 0.0;
     
      Float_t HT = 0.0;
      Float_t bjet1_pt = 0.0;
      Float_t dR_lepton_fwdjet = 0.0;
      Float_t dPhi_lepton_fwdjet = 0.0;
      Float_t dEta_lepton_fwdjet = 0.0;

      Float_t dR_lepton_bjet1 = 0.0;
      Float_t dPhi_lepton_bjet1 = 0.0;
      Float_t dEta_lepton_bjet1 = 0.0;
      Float_t bDiscriminatorValue = -2.0;
      Float_t bDiscriminatorValue_noBB = -2.0;

      int number_of_forward_jets = 0;
      int number_of_central_jets = 0;
      int number_of_jets = 0;
      
      Float_t dR_tH_forwardJet = 0.0;
      Float_t dR_bjet1_fwdj = 0.0;
      Float_t dR_leadpho_fwdjet = 0.0;
      Float_t dR_subleadpho_fwdjet = 0.0;
      Float_t dipho_pt = 0.0;
      
      Float_t deta_dipho_fwdj = 0.0;
      Float_t dR_leadPho_subleadPho = 0.0;
      Float_t dR_dipho_fwdjet = 0.0;
      Float_t num_bjets = 0.0;
      
      // repeat to match BDT variables
      Float_t n_jets = 0.0;
      Float_t fwdjet1_fabseta = 0.0;
      Float_t lepton_ch = 0;
      Float_t bjet_mult = 0;
      Float_t top_mass_transverse = 0.0;   
      Float_t bjet1_discr = 0.0;
      Float_t dR_lepton_fwdj = 0.0;
      Float_t n_central_jets = 0.0;
      Float_t dR_leadpho_fwdj = 0.0;
      Float_t dR_subleadpho_fwdj = 0.0;
      Float_t deta_lepton_fwdj = 0.0;
      Float_t recoMET_pt = 0.0;     
      Float_t n_forward_jets = 0.0;
      Float_t dipho_dR = 0.0;
      Float_t dR_dipho_fwdj = 0.0;
      
      float leadIDMVA_;
      float subleadIDMVA_;
      float MetPt_;
      float MetPhi_;
      
      struct particleinfo {
        float pt, eta, phi , other , w , another; //other : for photon id, for diphoton mass, for jets btagging vals
        unsigned short number;
        bool isSet;
        TLorentzVector lorentzVector_;
        std::map<std::string,float> info;
        particleinfo( double pt_=-999, double eta_=-999, double phi_=-999 , double other_= -999 , double W= 1.0 ) {
          pt = pt_;
          eta = eta_;
          phi = phi_;
          other = other_;
          w = W;
          number = 255;
          isSet = false;
          lorentzVector_.SetPtEtaPhiM(pt,eta,phi,other_);
        };
        
        void set(double pt_=-999, double eta_=-999, double phi_=-999 , double other_= -999 , double W= 1.0 , double Another= -999 ) {
          pt = pt_;
          eta = eta_;
          phi = phi_;
          other = other_;
          w = W;
          another = Another;
          isSet = true;
          lorentzVector_.SetPtEtaPhiM(pt,eta,phi,0.);
        };
        
        TLorentzVector LorentzVector() {
          return lorentzVector_;
        };
        
        void SetLorentzVector(TLorentzVector lorentzVector) {
	      lorentzVector_.SetPxPyPzE(lorentzVector.Px(),lorentzVector.Py(),lorentzVector.Pz(),lorentzVector.Energy());
        };
      };
        
      particleinfo lepton ,  eventshapes;
      particleinfo foxwolf1 ; // foxwolf2 , foxwolf1Met, foxwolf2Met ;
    
  }; // closing: class THQLeptonicTagProducer
  
  THQLeptonicTagProducer::THQLeptonicTagProducer( const ParameterSet &iConfig ) :
    processId_( iConfig.getParameter<string>("processId") ),
    diPhotonToken_( consumes<View<flashgg::DiPhotonCandidate> >( iConfig.getParameter<InputTag> ( "DiPhotonTag" ) ) ),
    inputTagJets_( iConfig.getParameter<std::vector<edm::InputTag> >( "inputTagJets" ) ),
    electronToken_( consumes<View<flashgg::Electron> >( iConfig.getParameter<InputTag>( "ElectronTag" ) ) ),
    muonToken_( consumes<View<flashgg::Muon> >( iConfig.getParameter<InputTag>( "MuonTag" ) ) ),
    mvaResultToken_( consumes<View<flashgg::DiPhotonMVAResult> >( iConfig.getParameter<InputTag> ( "MVAResultTag" ) ) ),
    vertexToken_( consumes<View<reco::Vertex> >( iConfig.getParameter<InputTag> ( "VertexTag" ) ) ),
    METToken_( consumes<View<flashgg::Met> >( iConfig.getParameter<InputTag> ( "METTag" ) ) ),
    genParticleToken_( consumes<View<reco::GenParticle> >( iConfig.getParameter<InputTag> ( "GenParticleTag" ) ) ),
    genJetToken_ ( consumes<View<reco::GenJet> >( iConfig.getParameter<InputTag> ( "GenJetTag" ) ) ),
    // genJetToken_( consumes<View<reco::GenJet> >( iConfig.getUntrackedParameter<InputTag> ( "GenJetTag", InputTag( "slimmedGenJets" ) ) ) ),
    weightToken_( consumes<vector<flashgg::PDFWeightObject> >( iConfig.getUntrackedParameter<InputTag>( "WeightTag", InputTag( "flashggPDFWeightObject" ) ) ) ),
    rhoTag_( consumes<double>( iConfig.getParameter<InputTag>( "rhoTag" ) ) ),
    systLabel_( iConfig.getParameter<string> ( "SystLabel" ) ),
    MVAMethod_    ( iConfig.getParameter<string> ( "MVAMethod"    ) ) {

    //if(processId_.find("thq") != std::string::npos or processId_.find("thw") != std::string::npos){
    //  token_lhe = consumes<LHEEventProduct>( InputTag( "externalLHEProducer" )  );
    //}

    double default_Zmass_ = 91.9;

    vector<double> default_electronEtaCuts_;
    default_electronEtaCuts_.push_back( 1.4442 );
    default_electronEtaCuts_.push_back( 1.566 );
    default_electronEtaCuts_.push_back( 2.5 );

    leptonEtaThreshold_ = iConfig.getParameter<double>( "leptonEtaThreshold" );
    leptonPtThreshold_ = iConfig.getParameter<double>( "leptonPtThreshold" );
    electronEtaThresholds_ = iConfig.getParameter<vector<double > >( "electronEtaThresholds");
    leadPhoOverMassThreshold_ = iConfig.getParameter<double>( "leadPhoOverMassThreshold" );
    subleadPhoOverMassThreshold_ = iConfig.getParameter<double>( "subleadPhoOverMassThreshold" );
    MVAThreshold_ = iConfig.getParameter<double>( "MVAThreshold" );
    deltaRLepPhoThreshold_ = iConfig.getParameter<double>( "deltaRLepPhoThreshold" );
    deltaRJetLepThreshold_ = iConfig.getParameter<double>( "deltaRJetLepThreshold" );
    jetsNumberThreshold_ = iConfig.getParameter<double>( "jetsNumberThreshold" );
    bjetsNumberThreshold_ = iConfig.getParameter<double>( "bjetsNumberThreshold" );
    jetPtThreshold_ = iConfig.getParameter<double>( "jetPtThreshold" );
    jetEtaThreshold_ = iConfig.getParameter<double>( "jetEtaThreshold" );

    deltaRJetLeadPhoThreshold_ = iConfig.getParameter<double>( "deltaRJetLeadPhoThreshold" );
    deltaRJetSubLeadPhoThreshold_ = iConfig.getParameter<double>( "deltaRJetSubLeadPhoThreshold" );

    electronEtaThresholds_ = iConfig.getUntrackedParameter<vector<double > >( "electronEtaCuts",default_electronEtaCuts_);
    bDiscriminator_ = iConfig.getParameter<vector<double > >( "bDiscriminator" );
    bTag_ = iConfig.getParameter<string>( "bTag" );

    muPFIsoSumRelThreshold_ = iConfig.getParameter<double>( "muPFIsoSumRelThreshold" );
    PhoMVAThreshold_ = iConfig.getParameter<double>( "PhoMVAThreshold" );
    DeltaRTrkElec_ = iConfig.getParameter<double>( "DeltaRTrkElec" );

    deltaRPhoElectronThreshold_ = iConfig.getParameter<double>( "deltaRPhoElectronThreshold" );
    Zmass_ = iConfig.getUntrackedParameter<double>( "Zmass_", default_Zmass_ );
    deltaMassElectronZThreshold_ = iConfig.getUntrackedParameter<double>( "deltaMassElectronZThreshold_" );

    MuonEtaCut_ = iConfig.getParameter<double>( "MuonEtaCut");
    MuonPtCut_ = iConfig.getParameter<double>( "MuonPtCut");
    MuonIsoCut_ = iConfig.getParameter<double>( "MuonIsoCut");
    MuonPhotonDrCut_ = iConfig.getParameter<double>( "MuonPhotonDrCut");
    
    MinNLep_ = iConfig.getParameter<int>( "MinNLep");
    LeptonsZMassCut_ = iConfig.getParameter<double>( "LeptonsZMassCut");
 
    thqLeptonicMVAweightfile_ = iConfig.getParameter<edm::FileInPath>( "thqleptonicMVAweightfile" );

    tthVstHDNNfile_ = iConfig.getParameter<edm::FileInPath>( "tthVstHDNNfile" );
    tthVstHDNN_global_mean_ = iConfig.getParameter<std::vector<double>>( "tthVstHDNN_global_mean" );
    tthVstHDNN_global_stddev_ = iConfig.getParameter<std::vector<double>>( "tthVstHDNN_global_stddev" );
    tthVstHDNN_object_mean_ = iConfig.getParameter<std::vector<double>>( "tthVstHDNN_object_mean" );
    tthVstHDNN_object_stddev_ = iConfig.getParameter<std::vector<double>>( "tthVstHDNN_object_stddev" );

    if (MVAMethod_ != ""){

      dnn_ttH_vs_tH = new TTH_DNN_Helper(tthVstHDNNfile_.fullPath());
      dnn_ttH_vs_tH->SetInputShapes(23, 9, 8);
      dnn_ttH_vs_tH->SetPreprocessingSchemes(tthVstHDNN_global_mean_, tthVstHDNN_global_stddev_, tthVstHDNN_object_mean_, tthVstHDNN_object_stddev_);
        
      thqLeptonicMva_.reset( new TMVA::Reader() );

      thqLeptonicMva_->AddVariable("n_jets", &n_jets);
      thqLeptonicMva_->AddVariable("fwdjet1_fabseta", &fwdjet1_fabseta);
      thqLeptonicMva_->AddVariable("lepton_ch", &lepton_ch);
      thqLeptonicMva_->AddVariable("bjet_mult", &bjet_mult); 
      thqLeptonicMva_->AddVariable("top_mass_transverse", &top_mass_transverse);   
      thqLeptonicMva_->AddVariable("bjet1_pt", &bjet1_pt);
      thqLeptonicMva_->AddVariable("dR_lepton_bjet1", &dR_lepton_bjet1);
      thqLeptonicMva_->AddVariable("bjet1_discr", &bjet1_discr);
      thqLeptonicMva_->AddVariable("dR_lepton_fwdj", &dR_lepton_fwdj);
      thqLeptonicMva_->AddVariable("n_central_jets", &n_central_jets);
      thqLeptonicMva_->AddVariable("dR_tH_forwardJet", &dR_tH_forwardJet);
      thqLeptonicMva_->AddVariable("dR_bjet1_fwdj", &dR_bjet1_fwdj);
      thqLeptonicMva_->AddVariable("dR_leadpho_fwdj", &dR_leadpho_fwdj);
      thqLeptonicMva_->AddVariable("dR_subleadpho_fwdj", &dR_subleadpho_fwdj);
      thqLeptonicMva_->AddVariable("dipho_pt", &dipho_pt);
      thqLeptonicMva_->AddVariable("HT", &HT);
      thqLeptonicMva_->AddVariable("deta_lepton_fwdj", &deta_lepton_fwdj);
      thqLeptonicMva_->AddVariable("recoMET_pt", &recoMET_pt);     
      thqLeptonicMva_->AddVariable("n_forward_jets", &n_forward_jets);
      thqLeptonicMva_->AddVariable("deta_dipho_fwdj", &deta_dipho_fwdj);
      thqLeptonicMva_->AddVariable("dipho_dR", &dipho_dR);
      thqLeptonicMva_->AddVariable("dR_dipho_fwdj", &dR_dipho_fwdj);
      
      thqLeptonicMva_->BookMVA("BDT","/afs/cern.ch/user/c/cbarrera/CMS/tH/CMSSW_10_5_0/src/flashgg/Taggers/data/COBOct2019_TMVAClassification_BDT.weights.xml");
    }

    for (unsigned i = 0 ; i < inputTagJets_.size() ; i++) {
        auto token = consumes<View<flashgg::Jet> >(inputTagJets_[i]);
        tokenJets_.push_back(token);
    }
    
    produces<vector<THQLeptonicTag> >();
    produces<vector<THQLeptonicTagTruth> >();

  } // closing: THQLeptonicTagProducer::THQLeptonicTagProducer
  
  
  /* ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
     ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
     ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
     ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
     ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
     ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
     ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
     ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
     ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- */

  void THQLeptonicTagProducer::produce( Event &evt, const EventSetup & ) {
    
    JetCollectionVector Jets( inputTagJets_.size() );
    
    for( size_t j = 0; j < inputTagJets_.size(); ++j ) {
      evt.getByToken( tokenJets_[j], Jets[j] );
    }

    edm::Handle<double>  rho;
    evt.getByToken(rhoTag_,rho);
    float rho_    = *rho;

    Handle<View<flashgg::DiPhotonCandidate> > diPhotons;
    evt.getByToken( diPhotonToken_, diPhotons );

    Handle<View<flashgg::Muon> > theMuons;
    evt.getByToken( muonToken_, theMuons );

    Handle<View<flashgg::Electron> > theElectrons;
    evt.getByToken( electronToken_, theElectrons );

    Handle<View<flashgg::DiPhotonMVAResult> > mvaResults;
    evt.getByToken( mvaResultToken_, mvaResults );

    std::unique_ptr<vector<THQLeptonicTag> > thqltags( new vector<THQLeptonicTag> );

    Handle<View<reco::Vertex> > vertices;
    evt.getByToken( vertexToken_, vertices );

    Handle<View<flashgg::Met> > METs;
    evt.getByToken( METToken_, METs );

    Handle<View<reco::GenParticle> > genParticles;
    Handle<View<reco::GenJet> > genJets;

    std::unique_ptr<vector<THQLeptonicTagTruth> > truths( new vector<THQLeptonicTagTruth> );
    Point higgsVtx;

    edm::RefProd<vector<THQLeptonicTagTruth> > rTagTruth = evt.getRefBeforePut<vector<THQLeptonicTagTruth> >();
    unsigned int idx = 0;

    assert( diPhotons->size() == mvaResults->size() );

    bool photonSelection = false;
    double idmva1 = 0.;
    double idmva2 = 0.;
    
    for( unsigned int diphoIndex = 0; diphoIndex < diPhotons->size(); diphoIndex++ ) {

        hasGoodElec = false; hasVetoElec = false;
        hasGoodMuons = false;
            
        unsigned int jetCollectionIndex = diPhotons->ptrAt( diphoIndex )->jetCollectionIndex();
            
        edm::Ptr<flashgg::DiPhotonCandidate> dipho = diPhotons->ptrAt( diphoIndex );
        edm::Ptr<flashgg::DiPhotonMVAResult> mvares = mvaResults->ptrAt( diphoIndex );

        flashgg::THQLeptonicTag thqltags_obj( dipho, mvares );

        if( ! evt.isRealData() ) {
          
            evt.getByToken( genParticleToken_, genParticles );
            evt.getByToken( genJetToken_, genJets );
            // thqltags_obj.setGenCollection(genParticles);
          
        }

        if( dipho->leadingPhoton()->pt() < ( dipho->mass() )*leadPhoOverMassThreshold_ ) { continue; }

        if( dipho->subLeadingPhoton()->pt() < ( dipho->mass() )*subleadPhoOverMassThreshold_ ) { continue; }

        idmva1 = dipho->leadingPhoton()->phoIdMvaDWrtVtx( dipho->vtx() );
        idmva2 = dipho->subLeadingPhoton()->phoIdMvaDWrtVtx( dipho->vtx() );

        if( idmva1 <= PhoMVAThreshold_ || idmva2 <= PhoMVAThreshold_ ) { continue; }

        // if( mvares->result < MVAThreshold_ ) { continue; } // @JAM deactivate as in TTH lep to add variables in TMVA

        photonSelection = true;
        
        G1.SetPtEtaPhiM( diPhotons->ptrAt( diphoIndex )->leadingPhoton()->pt(),
               diPhotons->ptrAt( diphoIndex )->leadingPhoton()->eta(),
               diPhotons->ptrAt( diphoIndex )->leadingPhoton()->phi() , 
                0 );
        particles_LorentzVector.push_back( G1 );
            
        G2.SetPtEtaPhiM( diPhotons->ptrAt( diphoIndex )->subLeadingPhoton()->pt(),
               diPhotons->ptrAt( diphoIndex )->subLeadingPhoton()->eta(),
               diPhotons->ptrAt( diphoIndex )->subLeadingPhoton()->phi(),
                0 );
        particles_LorentzVector.push_back(G2);

        particles_RhoEtaPhiVector.push_back( math::RhoEtaPhiVector(G1.Pt(), G1.Eta() , G1.Phi() ) );
        particles_RhoEtaPhiVector.push_back( math::RhoEtaPhiVector(G2.Pt(), G2.Eta() , G2.Phi() ) );
       
        if( METs->size() != 1 ) { std::cout << "WARNING - #MET is not 1" << std::endl;}
        Ptr<flashgg::Met> theMET = METs->ptrAt( 0 );
        thqltags_obj.setRECOMET(theMET);

        metL.SetPtEtaPhiE( theMET->getCorPt(),
             theMET->eta(),
             theMET->getCorPhi(),
             theMET->energy()
            ) ; 

      /* ---- ---- ---- ---- ---- ---- ---- ---- ---- --
         ---- ---- ---- ---- LEPTONS ---- ---- ---- ----
         ---- ---- ---- ---- ---- ---- ---- ---- ---- -- */      
            
      std::vector<edm::Ptr<flashgg::Electron>> goodElectrons = selectElectrons(theElectrons->ptrs(), dipho, leptonPtThreshold_, electronEtaThresholds_,
                                                                      deltaRPhoElectronThreshold_, deltaMassElectronZThreshold_, DeltaRTrkElec_, 0, 2);

      std::vector<edm::Ptr<flashgg::Muon>> goodMuons = selectMuons(theMuons->ptrs(), dipho, vertices->ptrs(), MuonPtCut_, MuonEtaCut_, MuonIsoCut_, 
                                                                   MuonPhotonDrCut_, 0, 2);
      
      std::vector<edm::Ptr<flashgg::Electron>> goodElectrons_tight = selectElectrons(theElectrons->ptrs(), dipho, leptonPtThreshold_, electronEtaThresholds_,
                                                                      deltaRPhoElectronThreshold_, deltaMassElectronZThreshold_, DeltaRTrkElec_, 0, 3);

      std::vector<edm::Ptr<flashgg::Muon>> goodMuons_tight = selectMuons(theMuons->ptrs(), dipho, vertices->ptrs(), MuonPtCut_, MuonEtaCut_, MuonIsoCut_, 
                                                                   MuonPhotonDrCut_, 0, 3);

      // -------> Add functionality here to veto Z, in case of two or more leptons (one loop for mu and one loop for e)
      if (goodMuons.size()>=2) {

          std::vector<edm::Ptr<flashgg::Muon>> Muons_0;
          Muons_0 = goodMuons;
          std::vector<int> badIndexes;

          for (unsigned int i=0; i<Muons_0.size(); ++i) {
              for (unsigned int j=i+1; j<Muons_0.size(); ++j) {
                  TLorentzVector l1, l2;
                  l1.SetPtEtaPhiE(Muons_0[i]->pt(), Muons_0[i]->eta(), Muons_0[i]->phi(), Muons_0[i]->energy());
                  l2.SetPtEtaPhiE(Muons_0[j]->pt(), Muons_0[j]->eta(), Muons_0[j]->phi(), Muons_0[j]->energy());

                  if (fabs((l1+l2).M() - 91.187) < LeptonsZMassCut_) {
                      badIndexes.push_back(i);
                      badIndexes.push_back(j);
                  }
              }
          }

          if (badIndexes.size()!=0) {
              goodMuons.clear();
              for (unsigned int i=0; i<Muons_0.size(); ++i) {
                  bool isBad = false;
                  for (unsigned int j=0; j<badIndexes.size(); ++j) {
                      if (badIndexes[j]==(int)i)
                          isBad = true;
                  }
                  if (!isBad) goodMuons.push_back(Muons_0[i]);
              }
          }
      }        

      if (goodElectrons.size()>=2) {
          std::vector<int> badIndexes;
          std::vector<edm::Ptr<flashgg::Electron> > Electrons_0;
          Electrons_0 = goodElectrons;
          for (unsigned int i=0; i<Electrons_0.size(); ++i) {
              for (unsigned int j=i+1; j<Electrons_0.size(); ++j) {                       
                  TLorentzVector l1, l2;
                  l1.SetPtEtaPhiE(Electrons_0[i]->pt(), Electrons_0[i]->eta(), Electrons_0[i]->phi(), Electrons_0[i]->energy());
                  l2.SetPtEtaPhiE(Electrons_0[j]->pt(), Electrons_0[j]->eta(), Electrons_0[j]->phi(), Electrons_0[j]->energy());

                  if (fabs((l1+l2).M() - 91.187) < LeptonsZMassCut_) {
                      badIndexes.push_back(i);
                      badIndexes.push_back(j);
                  }
              }
          }
          
          if (badIndexes.size()!=0) {
              goodElectrons.clear();

              for (unsigned int i=0; i<Electrons_0.size(); ++i) {
                  bool isBad = false;
                  for (unsigned int j=0; j<badIndexes.size(); ++j) {
                      if (badIndexes[j]==(int)i)
                          isBad = true;
                  }
                  if (!isBad) goodElectrons.push_back(Electrons_0[i]);
              }
          }
      }      
      
      if( (goodMuons.size() + goodElectrons.size()) < (unsigned) MinNLep_ ) continue;
    
      hasGoodElec = ( goodElectrons.size() >= 1 ); 
      hasGoodMuons = ( goodMuons.size() >= 1 );

      LeptonType = 0; // 1:electron, 2:muon
      
      if ( ( hasGoodMuons && !hasGoodElec ) ||
           ( hasGoodMuons && hasGoodElec && (goodMuons[0]->pt() >= goodElectrons[0]->pt()) ) ) {
          LeptonType = 2;
      }
      
      cout << " " << endl;
      
      for( unsigned int muonIndex = 0; muonIndex < goodMuons.size(); muonIndex++ ) {
                
          // cout << "goodMuons[i]->pt() = " << goodMuons[muonIndex]->pt() << endl;
          
          Ptr<flashgg::Muon> muon = goodMuons[muonIndex];
          thqltags_obj.includeWeights( *goodMuons[muonIndex] );
          lepton.set( muon->pt(),
            muon->eta() ,
            muon->phi() ,
            muon->energy(),
            1. ,
            muon->charge() );
          particles_LorentzVector.push_back(lepton.LorentzVector());
          particles_RhoEtaPhiVector.push_back( math::RhoEtaPhiVector( lepton.pt, lepton.eta, lepton.phi ) );
      } // end of muons loop

      if( ( !hasGoodMuons && hasGoodElec ) ||
           ( hasGoodMuons && hasGoodElec && (goodElectrons[0]->pt() > goodMuons[0]->pt()) )) {
           LeptonType = 1;
      }
      
      for( unsigned int ElectronIndex = 0; ElectronIndex < goodElectrons.size(); ElectronIndex++ ) {

          // cout << "goodElectrons[i]->pt() = " << goodElectrons[ElectronIndex]->pt() << endl;
          
          
          thqltags_obj.includeWeights( *goodElectrons[ElectronIndex] );
                
          Ptr<Electron> Electron = goodElectrons[ElectronIndex];
          lepton.set( Electron->pt(),
                      Electron->eta() ,
                      Electron->phi() ,
                      Electron->energy(),
                      1. ,
                      Electron->charge() );
          particles_LorentzVector.push_back(lepton.LorentzVector());
          particles_RhoEtaPhiVector.push_back( math::RhoEtaPhiVector( lepton.pt, lepton.eta, lepton.phi ) );
      } //end of electron loop      
      
      //if( LeptonType == 0 )
      //  continue;
    
      if(LeptonType==1){
          lepton_ch = goodElectrons[0]->charge();
      }else if(LeptonType==2){
          lepton_ch = goodMuons[0]->charge();
      }else{
          lepton_ch = -1000;
          
          if(goodMuons.size()>0)
              lepton_ch = goodMuons[0]->charge();
          
          if(goodElectrons.size()>0)
              lepton_ch = goodElectrons[0]->charge();
          
          if(goodMuons.size()>0 && goodElectrons.size()>0){
              
              if(goodElectrons[0]->pt() > goodMuons[0]->pt()){
                  lepton_ch = goodElectrons[0]->charge();
              }else{
                  lepton_ch = goodMuons[0]->charge();
              }
          }
      }
      
      number_of_jets = 0;
      number_of_central_jets = 0;
      number_of_forward_jets = 0;
      
      HT = 0.0;
      
      std::vector<float> bTags;
      std::vector<float> bTags_noBB;
      
      /* ---- ---- ---- ---- ---- ---- ---- ---- ---- --
         ---- ---- ---- ---- - JETS - ---- ---- ---- ---
         ---- ---- ---- ---- ---- ---- ---- ---- ---- -- */ 
      
      for( unsigned int candIndex_outer = 0; candIndex_outer < Jets[jetCollectionIndex]->size() ; candIndex_outer++ ) {
          
          edm::Ptr<flashgg::Jet> thejet = Jets[jetCollectionIndex]->ptrAt( candIndex_outer );

          //std::cout << "prin: "<< Jets[jetCollectionIndex]->size() << " "<<thejet->pt() << " "<< thejet->eta()<<" "<< thejet->phi()<< " "<< thejet->energy() <<std::endl;

          // if(!thejet->passesPuJetId(dipho)) { continue; }
          if(!thejet->passesJetID( flashgg::Loose)) { continue; } // @JAM as in TTHLep

          if(fabs(thejet->eta()) > jetEtaThreshold_) { continue; } // @JAM threshold same value as in TTHLep

          if(thejet->pt() < jetPtThreshold_) { continue; }

          float dRPhoLeadJet = deltaR( thejet->eta(), thejet->phi(), dipho->leadingPhoton()->superCluster()->eta(), dipho->leadingPhoton()->superCluster()->phi() ) ;
          float dRPhoSubLeadJet = deltaR( thejet->eta(), thejet->phi(), dipho->subLeadingPhoton()->superCluster()->eta(), dipho->subLeadingPhoton()->superCluster()->phi() );

          if( dRPhoLeadJet < deltaRJetLeadPhoThreshold_ || dRPhoSubLeadJet < deltaRJetSubLeadPhoThreshold_ ) { continue; }

          TLorentzVector jet_lorentzVector;
          jet_lorentzVector.SetPtEtaPhiE(  thejet->pt() , thejet->eta() , thejet->phi() , thejet->energy() );
          //std::cout <<  thejet->pt() << " "<< thejet->eta()<<" "<< thejet->phi()<< " "<< thejet->energy() <<std::endl;
          particles_LorentzVector.push_back( jet_lorentzVector );
          particles_RhoEtaPhiVector.push_back( math::RhoEtaPhiVector( thejet->pt(), thejet->eta(), thejet->phi() ) );
                
          double minDrLepton = 999.;
          for(auto mu : goodMuons) {
              float dRJetLepton = deltaR( thejet->eta(), thejet->phi(), mu->eta() , mu->phi() );
              if( dRJetLepton < minDrLepton ) { minDrLepton = dRJetLepton; }
          }
    
          for(auto ele : goodElectrons) {
              float dRJetLepton = deltaR( thejet->eta(), thejet->phi(), ele->eta() , ele->phi() );
              if( dRJetLepton < minDrLepton ) { minDrLepton = dRJetLepton; }
          }

          if( minDrLepton < deltaRJetLepThreshold_) continue;

          if( abs( thejet->eta() ) > 1.0 )
              number_of_forward_jets ++;
          else
              number_of_central_jets ++;
    
          number_of_jets++;
            
          if(bTag_ == "pfDeepCSV") bDiscriminatorValue = thejet->bDiscriminator("pfDeepCSVJetTags:probb")+thejet->bDiscriminator("pfDeepCSVJetTags:probbb") ;
          else  bDiscriminatorValue = thejet->bDiscriminator( bTag_ );
                    
          if(bTag_ == "pfDeepCSV") bDiscriminatorValue_noBB = thejet->bDiscriminator("pfDeepCSVJetTags:probb");
          else  bDiscriminatorValue_noBB = thejet->bDiscriminator( bTag_ );
          
        
          bDiscriminatorValue >= 0. ? bTags.push_back(bDiscriminatorValue) : bTags.push_back(-1.);
          bDiscriminatorValue_noBB >= 0. ? bTags_noBB.push_back(bDiscriminatorValue_noBB) : bTags_noBB.push_back(-1.);

          if( bDiscriminatorValue > bDiscriminator_[0] ) {
              LooseBJetVect_PtSorted.push_back( thejet ); 
              LooseBJetVect.push_back( thejet );
          }

          if( bDiscriminatorValue > bDiscriminator_[1] ) {
              MediumBJetVect.push_back( thejet ); 
              MediumBJetVect_PtSorted.push_back( thejet );
          }

          if( bDiscriminatorValue > bDiscriminator_[2] ) {
              TightBJetVect_PtSorted.push_back( thejet ); 
              TightBJetVect.push_back( thejet );
          }

          HT+=thejet->pt();
          SelJetVect.push_back( thejet ); 
          SelJetVect_EtaSorted.push_back( thejet );
          SelJetVect_PtSorted.push_back( thejet );
          SelJetVect_BSorted.push_back( thejet );
      } // end of jets loop

      //if( !( ( number_of_central_jets == 2 && number_of_forward_jets == 0 ) || 
      //       ( number_of_central_jets == 2 && number_of_forward_jets == 1 ) || 
      //       ( number_of_central_jets == 3 && number_of_forward_jets == 1 ) )
      //) continue;

      // COB 14.Feb.2019
      // if ( !(number_of_central_jets > 0 && number_of_forward_jets > 0) ) continue;
      
      if(number_of_jets < 2.0) continue;      

      thqltags_obj.nCentral_Jets = number_of_central_jets;
      thqltags_obj.nForward_Jets = number_of_forward_jets;

      //Calculate scalar sum of jets
      thqltags_obj.setHT(HT);
      
      std::sort(LooseBJetVect_PtSorted.begin(),LooseBJetVect_PtSorted.end(),GreaterByPt());       
      std::sort(LooseBJetVect.begin(),LooseBJetVect.end(), GreaterByBTagging("pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb"));
      

      std::sort(MediumBJetVect_PtSorted.begin(),MediumBJetVect_PtSorted.end(),GreaterByPt()); 
      std::sort(MediumBJetVect.begin(),MediumBJetVect.end(), GreaterByBTagging("pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb")); 

      std::sort(TightBJetVect_PtSorted.begin(),TightBJetVect_PtSorted.end(),GreaterByPt()); 
      std::sort(TightBJetVect.begin(),TightBJetVect.end(), GreaterByBTagging("pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb")); 

      std::sort(SelJetVect_EtaSorted.begin(),SelJetVect_EtaSorted.end(),GreaterByEta()); 
      std::sort(SelJetVect_PtSorted.begin(),SelJetVect_PtSorted.end(),GreaterByPt()); 
      std::sort(SelJetVect_BSorted.begin(),SelJetVect_BSorted.end(), GreaterByBTagging("pfDeepCSVJetTags:probb", "pfDeepCSVJetTags:probbb")); 

      if(bTag_ == "pfDeepCSV") bDiscriminatorValue =  SelJetVect_BSorted.at(0)->bDiscriminator("pfDeepCSVJetTags:probb")+ SelJetVect_BSorted.at(0)->bDiscriminator("pfDeepCSVJetTags:probbb") ;
      else  bDiscriminatorValue =  SelJetVect_BSorted.at(0)->bDiscriminator( bTag_ );
      
      // cout << "SelJetVect_BSorted.at(0)->bDiscriminator(bTag_.c_str()) = " <<
      // bDiscriminatorValue << endl;
      
      // @JAM 13.March.2019
      if (bDiscriminatorValue < bDiscriminator_[0]) continue; 
      
      // forward jet - bjet overlap removal
      if((SelJetVect_BSorted.at(0)->eta() == SelJetVect_EtaSorted.at(0)->eta()) && (SelJetVect_BSorted.at(0)->phi() == SelJetVect_EtaSorted.at(0)->phi())){
          
          // cout << "removed forward ... " << endl;    
          // cout << "SelJetVect_BSorted.at(0)->eta() = " << SelJetVect_BSorted.at(0)->eta() << endl; 
          // cout << "SelJetVect_EtaSorted.at(0)->eta() = " << SelJetVect_EtaSorted.at(0)->eta() << endl;
          
          SelJetVect_EtaSorted.erase(SelJetVect_EtaSorted.begin());
      }
      
      // @JAM 20.March.2019 b-jet multiplicity
      num_bjets=0;
      double bDiscriminatorValue_mult = -2.0;
      
      for(unsigned int p =0; p<SelJetVect_BSorted.size(); p++){
          
          bDiscriminatorValue_mult =  SelJetVect_BSorted.at(p)->bDiscriminator("pfDeepCSVJetTags:probb")+ SelJetVect_BSorted.at(p)->bDiscriminator("pfDeepCSVJetTags:probbb") ;
          
          if (bDiscriminatorValue_mult < bDiscriminator_[0])
              num_bjets++;
      }
      
      thqltags_obj.setBjetMult(num_bjets);
      
      bjetL.SetPtEtaPhiE(SelJetVect_BSorted.at(0)->pt(), SelJetVect_BSorted.at(0)->eta(), SelJetVect_BSorted.at(0)->phi(), SelJetVect_BSorted.at(0)->energy());
      tH_recoL = G1 + G2 + bjetL + metL + lepton.LorentzVector();
      
      dR_tH_forwardJet = deltaR( tH_recoL.Eta() , tH_recoL.Phi() , SelJetVect_EtaSorted.at(0)->eta() , SelJetVect_EtaSorted.at(0)->phi() );
            
      thqltags_obj.dRtHsystem_fwdjet = deltaR( tH_recoL.Eta() , tH_recoL.Phi() , SelJetVect_EtaSorted.at(0)->eta() , SelJetVect_EtaSorted.at(0)->phi() );

      leadIDMVA_ = dipho->leadingPhoton()->phoIdMvaDWrtVtx( dipho->vtx() );
      subleadIDMVA_ = dipho->subLeadingPhoton()->phoIdMvaDWrtVtx( dipho->vtx() );
      MetPt_ = METs->ptrAt( 0 ) -> getCorPt();
      MetPhi_ = METs->ptrAt( 0 ) -> phi();
      
      std::vector<double> global_features_ttH_vs_tH;
      global_features_ttH_vs_tH.resize(23);
      global_features_ttH_vs_tH[0] = dipho->leadingPhoton()->eta();
      global_features_ttH_vs_tH[1] = dipho->subLeadingPhoton()->eta();
      global_features_ttH_vs_tH[2] = dipho->leadingPhoton()->phi();
      global_features_ttH_vs_tH[3] = dipho->subLeadingPhoton()->phi();
      global_features_ttH_vs_tH[4] = dipho->leadingPhoton()->pt()/dipho->mass();
      global_features_ttH_vs_tH[5] = dipho->subLeadingPhoton()->pt()/dipho->mass();
      global_features_ttH_vs_tH[6] = TMath::Max( leadIDMVA_, subleadIDMVA_);
      global_features_ttH_vs_tH[7] = TMath::Min( leadIDMVA_, subleadIDMVA_);
      global_features_ttH_vs_tH[8] = log(MetPt_);
      global_features_ttH_vs_tH[9] = MetPhi_;
      global_features_ttH_vs_tH[10] = dipho->leadingPhoton()->hasPixelSeed();
      global_features_ttH_vs_tH[11] = dipho->subLeadingPhoton()->hasPixelSeed();
      global_features_ttH_vs_tH[12] = dipho->rapidity();
      global_features_ttH_vs_tH[13] = dipho->pt()/dipho->mass();
      global_features_ttH_vs_tH[14] = deltaR( dipho->leadingPhoton()->eta(),dipho->leadingPhoton()->phi(), dipho->subLeadingPhoton()->eta(),dipho->subLeadingPhoton()->phi());
      global_features_ttH_vs_tH[15] = bTags_noBB.size() > 0 ? bTags_noBB[0] : -1.;
      global_features_ttH_vs_tH[16] = bTags_noBB.size() > 1 ? bTags_noBB[1]: -1.;
      global_features_ttH_vs_tH[17] = number_of_jets;
      global_features_ttH_vs_tH[18] = float(goodMuons_tight.size() + goodElectrons_tight.size());
      
      // SelJetVect_EtaSorted.at(0)->eta()
      // SelJetVect_EtaSorted.at(0)->pt()
      
      // cout << "SelJetVect_EtaSorted.at(0)->eta() = " << SelJetVect_EtaSorted.at(0)->eta() << endl;
      // cout << "SelJetVect_EtaSorted.at(0)->pt() = " << SelJetVect_EtaSorted.at(0)->pt() << endl;

      double lep1_charge, lep2_charge;
      calculate_lepton_charges(lep1_charge, lep2_charge, goodMuons, goodElectrons);
      
      global_features_ttH_vs_tH[19] = lep1_charge;
      global_features_ttH_vs_tH[20] = lep2_charge;
      global_features_ttH_vs_tH[21] = SelJetVect_EtaSorted.at(0)->eta();
      global_features_ttH_vs_tH[22] = SelJetVect_EtaSorted.at(0)->pt(); 
      
      dnn_ttH_vs_tH->SetInputs(SelJetVect_PtSorted, goodMuons, goodElectrons, global_features_ttH_vs_tH);
      ttH_vs_tH_dnn_score = dnn_ttH_vs_tH->EvaluateDNN();

      if( photonSelection ) {
          
          // cout << " ttH_vs_tH_dnn_score = " << ttH_vs_tH_dnn_score << endl;
      
          thqltags_obj.dnn_SD = ttH_vs_tH_dnn_score;

          // TTH DNN variables
          thqltags_obj.tag_dipho_leadingPhoton_eta = dipho->leadingPhoton()->eta();
          thqltags_obj.tag_subLeadingPhoton_eta = dipho->subLeadingPhoton()->eta();
          thqltags_obj.tag_dipho_leadingPhoton_phi = dipho->leadingPhoton()->phi();
          thqltags_obj.tag_dipho_subLeadingPhoton_phi = dipho->subLeadingPhoton()->phi();
          thqltags_obj.tag_dipho_leadingPhoton_pt_dipho_mass = dipho->leadingPhoton()->pt()/dipho->mass();
          thqltags_obj.tag_dipho_subLeadingPhoton_pt_dipho_mass = dipho->subLeadingPhoton()->pt()/dipho->mass();
          thqltags_obj.tag_MaxIDMVA = TMath::Max( leadIDMVA_, subleadIDMVA_);
          thqltags_obj.tag_MinIDMVA = TMath::Min( leadIDMVA_, subleadIDMVA_);
          thqltags_obj.tag_logMetPt = log(MetPt_);
          thqltags_obj.tag_MetPhi = MetPhi_;
          thqltags_obj.tag_dipho_leadingPhoton_hasPixelSeed = dipho->leadingPhoton()->hasPixelSeed();
          thqltags_obj.tag_dipho_subLeadingPhoton_hasPixelSeed = dipho->subLeadingPhoton()->hasPixelSeed();
          thqltags_obj.tag_dipho_rapidity = dipho->rapidity();
          thqltags_obj.tag_dipho_pt_dipho_mass = dipho->pt()/dipho->mass();
          thqltags_obj.tag_deltaR_dipho_leadingPhoton_subLeadingPhoton = deltaR( dipho->leadingPhoton()->eta(),dipho->leadingPhoton()->phi(), dipho->subLeadingPhoton()->eta(),dipho->subLeadingPhoton()->phi());
          thqltags_obj.tag_bTags_noBB_size_0 = bTags_noBB.size() > 0 ? bTags_noBB[0] : -1.;
          thqltags_obj.tag_bTags_noBB_size_1 = bTags_noBB.size() > 1 ? bTags_noBB[1]: -1.;
          thqltags_obj.tag_number_of_jets = number_of_jets;
          thqltags_obj.tag_float_tight_leptons = float(goodMuons_tight.size() + goodElectrons_tight.size());
      
          thqltags_obj.num_medium_leptons = goodMuons.size() + goodElectrons.size();
          
          //&& ( ( (tagMuons.size() == 1 && muonJets) and  (tagElectrons.size() == 0 && !ElectronJets) )  || ( (tagMuons.size() == 0 && !muonJets)  and  (tagElectrons.size() == 1 && ElectronJets) ) ) ) 
                
          EventShapeVariables shapeVars(particles_RhoEtaPhiVector);
          //std::cout  << "aplanarity: "<<shapeVars.aplanarity()<<std::endl;
          eventshapes.set( shapeVars.aplanarity(),
              shapeVars.C(),
              shapeVars.circularity(),
              shapeVars.D(),
              shapeVars.isotropy(),
              shapeVars.sphericity());

        thqltags_obj.setrho(rho_);

        thqltags_obj.setLeptonType(LeptonType);
    
        thqltags_obj.includeWeights( *dipho );

        thqltags_obj.photonWeights = dipho->leadingPhoton()->centralWeight()*dipho->subLeadingPhoton()->centralWeight() ;

        thqltags_obj.setJets( SelJetVect_PtSorted , SelJetVect_EtaSorted);
        thqltags_obj.setBJets( SelJetVect_BSorted );

        thqltags_obj.bTagWeight = 1.0;
        thqltags_obj.bTagWeightDown = 1.0;
        thqltags_obj.bTagWeightUp = 1.0;

        
        for( auto j : SelJetVect_PtSorted ){
            thqltags_obj.includeWeights( *j );
    
            if( j->hasWeight("JetBTagCutWeightCentral") ){
                thqltags_obj.bTagWeight *= j->weight( "JetBTagCutWeightCentral" );
                thqltags_obj.bTagWeightDown *= j->weight( "JetBTagCutWeightDown01sigma" );
                thqltags_obj.bTagWeightUp *= j->weight( "JetBTagCutWeightUp01sigma" );
            } else {
                cout << "BTag weight is not set in jet" << endl;
            }
        }

        thqltags_obj.setVertices( vertices->ptrs() );

        std::vector <float> a; std::vector <float> b; std::vector <float> c; std::vector <float> d;

        thqltags_obj.setLeptonVertices( "muon", a, b, c, d) ;

        thqltags_obj.setMuons( goodMuons );
           
        thqltags_obj.setLeptonVertices( "electron", a, b, c, d) ;
   
        thqltags_obj.setElectrons( goodElectrons );        
        thqltags_obj.setDiPhotonIndex( diphoIndex );
        thqltags_obj.setSystLabel( systLabel_ );

        //thqltags_obj.setFoxAndAplanarity( foxwolf1.another , eventshapes.pt );
        thqltags_obj.setMETPtEtaPhiE( "SolvedMET", metW_check.Pt(), metW_check.Eta(), metW_check.Phi(), metW_check.E() );

        //------------------------------------------------- 
        //Angular distributions between different objects

        if (LeptonType == 1) {

            float tmp1 = deltaR(goodElectrons[0]->eta(),goodElectrons[0]->phi(),
                              SelJetVect_EtaSorted.at(0)->eta(),SelJetVect_EtaSorted.at(0)->phi());
            float tmp2 = deltaPhi(goodElectrons[0]->phi(),SelJetVect_EtaSorted.at(0)->phi());
            float tmp3 = abs(goodElectrons[0]->eta() - SelJetVect_EtaSorted.at(0)->eta());
            dR_lepton_fwdjet = tmp1;
            dPhi_lepton_fwdjet = tmp2;
            dEta_lepton_fwdjet = tmp3;

            float tmp4 = deltaR(goodElectrons[0]->eta(),goodElectrons[0]->phi(),
                              SelJetVect_BSorted.at(0)->eta(),SelJetVect_BSorted.at(0)->phi());
            float tmp5 = deltaPhi(goodElectrons[0]->phi(),SelJetVect_BSorted.at(0)->phi());
            float tmp6 = abs(goodElectrons[0]->eta() - SelJetVect_BSorted.at(0)->eta());
            dR_lepton_bjet1 = tmp4;
            dPhi_lepton_bjet1 = tmp5;
            dEta_lepton_bjet1 = tmp6;

        } else { 

            float tmp1 = deltaR(goodMuons[0]->eta(),goodMuons[0]->phi(),
                              SelJetVect_EtaSorted.at(0)->eta(),SelJetVect_EtaSorted.at(0)->phi());
            float tmp2 = deltaPhi(goodMuons[0]->phi(),SelJetVect_EtaSorted.at(0)->phi());
            float tmp3 = abs(goodMuons[0]->eta() - SelJetVect_EtaSorted.at(0)->eta());
            dR_lepton_fwdjet = tmp1;
            dPhi_lepton_fwdjet = tmp2;
            dEta_lepton_fwdjet = tmp3;

            float tmp4 = deltaR(goodMuons[0]->eta(),goodMuons[0]->phi(),
                              SelJetVect_BSorted.at(0)->eta(),SelJetVect_BSorted.at(0)->phi());
            float tmp5 = deltaPhi(goodMuons[0]->phi(),SelJetVect_BSorted.at(0)->phi());
            float tmp6 = abs(goodMuons[0]->eta() - SelJetVect_BSorted.at(0)->eta());
            dR_lepton_bjet1 = tmp4;
            dPhi_lepton_bjet1 = tmp5;
            dEta_lepton_bjet1 = tmp6;
        }
    
        topReco( &SelJetVect_BSorted );
        
        thqltags_obj.setBDT_value(thqLeptonicMvaResult_value_);
        
        thqltags_obj.setTopMass_BSorted(topMass); 
        thqltags_obj.setTopTransverseMass(topMassTransverse);
        
        dR_leadPho_subleadPho = deltaR(dipho->leadingPhoton()->eta(),dipho->leadingPhoton()->phi(),
                                             dipho->subLeadingPhoton()->eta(),dipho->subLeadingPhoton()->phi());
        float dPhi_leadPho_subleadPho = deltaPhi(dipho->leadingPhoton()->phi(),dipho->subLeadingPhoton()->phi()); 

        dR_dipho_fwdjet = deltaR(dipho->eta(),dipho->phi(),SelJetVect_EtaSorted.at(0)->eta(),SelJetVect_EtaSorted.at(0)->phi());
        float dPhi_dipho_fwdjet = deltaPhi(dipho->phi(),SelJetVect_EtaSorted.at(0)->phi());

        dR_leadpho_fwdjet = deltaR(dipho->leadingPhoton()->eta(),dipho->leadingPhoton()->phi(),
                                         SelJetVect_EtaSorted.at(0)->eta(),SelJetVect_EtaSorted.at(0)->phi());
        dR_subleadpho_fwdjet = deltaR(dipho->subLeadingPhoton()->eta(),dipho->subLeadingPhoton()->phi(),
                                            SelJetVect_EtaSorted.at(0)->eta(),SelJetVect_EtaSorted.at(0)->phi());

        dR_bjet1_fwdj = deltaR(SelJetVect_BSorted.at(0)->eta(),SelJetVect_BSorted.at(0)->phi(),
                                     SelJetVect_EtaSorted.at(0)->eta(),SelJetVect_EtaSorted.at(0)->phi());
        
        deta_dipho_fwdj = fabs(dipho->eta()-SelJetVect_EtaSorted.at(0)->eta());
        
        dipho_pt = dipho->pt();
                
        thqltags_obj.setDeltaR_Photons(dR_leadPho_subleadPho); 
        thqltags_obj.setDeltaPhi_Photons(dPhi_leadPho_subleadPho);
        thqltags_obj.setDeltaR_dipho_fwdj(dR_dipho_fwdjet);
        thqltags_obj.setDeltaR_lepton_fwdj(dR_lepton_fwdjet);
        thqltags_obj.setDeltaPhi_dipho_fwdj(dPhi_dipho_fwdjet);
        thqltags_obj.setDeltaPhi_lepton_fwdj(dPhi_lepton_fwdjet);
        thqltags_obj.setDeltaEta_lepton_fwdj(dEta_lepton_fwdjet);

        thqltags_obj.setDeltaR_lepton_bjet1(dR_lepton_bjet1);
        thqltags_obj.setDeltaEta_lepton_bjet1(dEta_lepton_bjet1);
        thqltags_obj.setDeltaPhi_lepton_bjet1(dPhi_lepton_bjet1);

        thqltags_obj.setDeltaR_leadpho_fwdj(dR_leadpho_fwdjet);
        thqltags_obj.setDeltaR_subleadpho_fwdj(dR_subleadpho_fwdjet);

        thqltags_obj.setDeltaR_bjet1_fwdj(dR_bjet1_fwdj);
        
        topReco( &MediumBJetVect_PtSorted );
        //thqltags_obj.setMVAres("Medium" ,  thqLeptonicMvaResult_value_ , topMass , fwdJet , bJet);
        thqltags_obj.nMedium_bJets = MediumBJetVect_PtSorted.size();

        topReco( &LooseBJetVect_PtSorted );
        //thqltags_obj.setMVAres("Loose" ,  thqLeptonicMvaResult_value_ , topMass , fwdJet , bJet);
        thqltags_obj.nLoose_bJets = LooseBJetVect_PtSorted.size();

        topReco( &TightBJetVect_PtSorted );
        //thqltags_obj.setMVAres("Tight" ,  thqLeptonicMvaResult_value_ , topMass , fwdJet , bJet);
        thqltags_obj.nTight_bJets = TightBJetVect_PtSorted.size();

        //cout << evt.isRealData() << endl;    
        
        if( ! evt.isRealData() ) {

            if(processId_.find("thq") != std::string::npos or processId_.find("thw") != std::string::npos){
              
                // ...

            }else if(processId_.find("h_") != std::string::npos or processId_.find("vbf") != std::string::npos){
                //temporary solution till ctcv issue on PDFWeightObject is solved :(
                Handle<vector<flashgg::PDFWeightObject> > WeightHandle;
                evt.getByToken( weightToken_, WeightHandle );
    
                for( unsigned int weight_index = 0; weight_index < (*WeightHandle).size(); weight_index++ ){
                    vector<uint16_t> compressed_weights = (*WeightHandle)[weight_index].pdf_weight_container;
                    std::vector<float> uncompressed = (*WeightHandle)[weight_index].uncompress( compressed_weights );
                    vector<uint16_t> compressed_alpha = (*WeightHandle)[weight_index].alpha_s_container;
                    std::vector<float> uncompressed_alpha = (*WeightHandle)[weight_index].uncompress( compressed_alpha );
                    vector<uint16_t> compressed_scale = (*WeightHandle)[weight_index].qcd_scale_container;
                    std::vector<float> uncompressed_scale = (*WeightHandle)[weight_index].uncompress( compressed_scale );
                    vector<uint16_t> compressed_nloweights = (*WeightHandle)[weight_index].pdfnlo_weight_container;
                    std::vector<float> uncompressed_nloweights = (*WeightHandle)[weight_index].uncompress( compressed_nloweights );

                    float central_w = uncompressed_scale[0];
    
                    for( unsigned int j=0; j<(*WeightHandle)[weight_index].pdf_weight_container.size();j++ ) {
                        thqltags_obj.setPdf(j,uncompressed[j]/ central_w );
                    }
        
                    if(uncompressed_alpha.size()>1){
                        thqltags_obj.setAlphaUp(uncompressed_alpha[0]/central_w );
                        thqltags_obj.setAlphaDown(uncompressed_alpha[1]/ central_w );
                    }else
                        thqltags_obj.setAlphaDown(uncompressed_alpha[0]/ central_w );

                    for( uint i = 1 ; i < 9 ; i ++ )
                        thqltags_obj.setScale(i-1,uncompressed_scale[i]/central_w );

                    if(uncompressed_nloweights.size()>0)
                        thqltags_obj.setPdfNLO(uncompressed_nloweights[0]/ central_w);
                    }
                }//end of reading PDF weights from PDFWeightObject

                //if( ! evt.isRealData() ) {
                //    evt.getByToken( genJetToken_, genJets );
                //}
          
               THQLeptonicTagTruth truth_obj;
               truth_obj.setDiPhoton ( dipho ); 
          
               for(unsigned int genLoop = 0 ; genLoop < genParticles->size(); genLoop++){
                   edm::Ptr<reco::GenParticle> partt = genParticles->ptrAt( genLoop );
                   //cout << genLoop << "  ID  " << partt->pdgId() << "  status "  << partt->status() << " mother " << partt->mother() << endl;
  
                   int pdgid = genParticles->ptrAt( genLoop )->pdgId();
                   if( pdgid == 25 || pdgid == 22 ) {
                       higgsVtx = genParticles->ptrAt( genLoop )->vertex();
                       break;
                   }
                }  
    
                truth_obj.setGenPV( higgsVtx );

                // --------
                //gen met
                TLorentzVector nu_lorentzVector, allnus_LorentzVector, promptnus_LorentzVector;
    
                for( unsigned int genLoop = 0 ; genLoop < genParticles->size(); genLoop++ ) {
                    edm::Ptr<reco::GenParticle> part = genParticles->ptrAt( genLoop );
                    bool fid_cut = (abs(part->eta())<5.0 && part->status()==1) ? 1 : 0;
                    bool isNu = (abs(part->pdgId())==12 || part->pdgId()==14 || part->pdgId()==16) ? 1 : 0;
                    if (!fid_cut || !isNu) continue;
                    if( part->isPromptFinalState() || part->isDirectPromptTauDecayProductFinalState()) {
                        nu_lorentzVector.SetPtEtaPhiE(  part->pt() , part->eta() , part->phi() , part->energy() );
                        promptnus_LorentzVector+=nu_lorentzVector;
                    }else{
                        nu_lorentzVector.SetPtEtaPhiE(  part->pt() , part->eta() , part->phi() , part->energy() );
                        allnus_LorentzVector+=nu_lorentzVector;
                    }
                }
          
                thqltags_obj.setMETPtEtaPhiE( "allPromptNus", promptnus_LorentzVector.Pt(), promptnus_LorentzVector.Eta(),
                promptnus_LorentzVector.Phi(),
                promptnus_LorentzVector.Energy() );
                thqltags_obj.setMETPtEtaPhiE( "allNus", allnus_LorentzVector.Pt(), allnus_LorentzVector.Eta(), allnus_LorentzVector.Phi(), allnus_LorentzVector.Energy() );
                thqltags_obj.setMETPtEtaPhiE( "genMetTrue", theMET->genMET()->pt(), theMET->genMET()->eta(), theMET->genMET()->phi(), theMET->genMET()->energy() );

                if(SelJetVect_PtSorted.size() > 1){
                    unsigned int index_leadq       = std::numeric_limits<unsigned int>::max();
                    unsigned int index_subleadq    = std::numeric_limits<unsigned int>::max();
                    unsigned int index_subsubleadq    = std::numeric_limits<unsigned int>::max();
                    float pt_leadq = 0., pt_subleadq = 0., pt_subsubleadq = 0.;

                    //Partons
                    for(unsigned int genLoop = 0 ; genLoop < genParticles->size(); genLoop++ ){
                        edm::Ptr<reco::GenParticle> part = genParticles->ptrAt( genLoop );
                        if(part->isHardProcess()){
                            if(abs( part->pdgId() ) <= 5 ){
                                if( part->pt() > pt_leadq ){
                                    index_subleadq = index_leadq;
                                    pt_subleadq = pt_leadq;
                                    index_leadq = genLoop;
                                    pt_leadq = part->pt();
                                }else if( part->pt() > pt_subleadq){
                                    index_subsubleadq  = index_subleadq;
                                    pt_subsubleadq  = pt_subleadq;
                                    index_subleadq = genLoop;
                                    pt_subleadq  = part->pt();
                                }else if( part->pt() > pt_subsubleadq ){
                                    index_subsubleadq = genLoop;
                                    pt_subleadq  = part->pt();
                                }
                            }
                        }
                    }
            
                    if( index_leadq < std::numeric_limits<unsigned int>::max() ) { truth_obj.setLeadingParton( genParticles->ptrAt( index_leadq ) ); }
                    if( index_subleadq < std::numeric_limits<unsigned int>::max() ) { truth_obj.setSubLeadingParton( genParticles->ptrAt( index_subleadq ) ); }
                    if( index_subsubleadq < std::numeric_limits<unsigned int>::max()) { truth_obj.setSubSubLeadingParton( genParticles->ptrAt( index_subsubleadq ));}
    
                    unsigned int index_gp_leadjet = std::numeric_limits<unsigned int>::max();
                    unsigned int index_gp_subleadjet = std::numeric_limits<unsigned int>::max();
                    unsigned int index_gp_leadphoton = std::numeric_limits<unsigned int>::max();
                    unsigned int index_gp_subleadphoton = std::numeric_limits<unsigned int>::max();
                    unsigned int index_gp_leadmuon = std::numeric_limits<unsigned int>::max();
                    unsigned int index_gp_subleadmuon = std::numeric_limits<unsigned int>::max();
                    unsigned int index_gp_leadelectron = std::numeric_limits<unsigned int>::max();
                    unsigned int index_gp_subleadelectron = std::numeric_limits<unsigned int>::max();
          
                    float dr_gp_leadjet = 999.;
                    float dr_gp_subleadjet = 999.;
                    float dr_gp_leadphoton = 999.;
                    float dr_gp_subleadphoton = 999.;
                    float dr_gp_leadmuon = 999.;
                    float dr_gp_subleadmuon = 999.;
                    float dr_gp_leadelectron = 999.;
                    float dr_gp_subleadelectron = 999.;
            
                    if (SelJetVect_PtSorted.size()>0)truth_obj.setLeadingJet( SelJetVect_PtSorted[0] );
                    if (SelJetVect_PtSorted.size()>1)truth_obj.setSubLeadingJet( SelJetVect_PtSorted[1] );
                    if (SelJetVect_PtSorted.size()>2)truth_obj.setSubSubLeadingJet( SelJetVect_PtSorted[2] );
                    if (SelJetVect_PtSorted.size()>0)truth_obj.setLeadingJet( SelJetVect_PtSorted[0] );
                    if (SelJetVect_PtSorted.size()>1)truth_obj.setSubLeadingJet( SelJetVect_PtSorted[1] );
                    if (thqltags_obj.muons().size()>0)truth_obj.setLeadingMuon( thqltags_obj.muons()[0] );
                    if (thqltags_obj.muons().size()>1)truth_obj.setSubLeadingMuon( thqltags_obj.muons()[1] );
                    if (thqltags_obj.electrons().size()>0)truth_obj.setLeadingElectron( thqltags_obj.electrons()[0] );
                    if (thqltags_obj.electrons().size()>1)truth_obj.setSubLeadingElectron( thqltags_obj.electrons()[1] );
            
                    //GEN-RECO Level Matching
                    for( unsigned int genLoop = 0 ; genLoop < genParticles->size(); genLoop++ ) {
                        edm::Ptr<reco::GenParticle> part = genParticles->ptrAt( genLoop );
                        if( part->isHardProcess()) {
                            float dr;
                            if (truth_obj.hasLeadingJet()) {
                                dr = deltaR( truth_obj.leadingJet()->eta(), truth_obj.leadingJet()->phi(), part->eta(), part->phi() );
                                if( dr < dr_gp_leadjet ) {
                                    dr_gp_leadjet = dr;
                                    index_gp_leadjet = genLoop;
                                }
                            }
        
                            if (truth_obj.hasSubLeadingJet()) {
                                dr = deltaR( truth_obj.subLeadingJet()->eta(), truth_obj.subLeadingJet()->phi(), part->eta(), part->phi() );
                               if( dr < dr_gp_subleadjet ) {
                                   dr_gp_subleadjet = dr;
                                   index_gp_subleadjet = genLoop;
                               }
                            }
        
                            if (truth_obj.hasDiPhoton()) {
                                dr = deltaR( truth_obj.diPhoton()->leadingPhoton()->eta(), truth_obj.diPhoton()->leadingPhoton()->phi(), part->eta(), part->phi() );
                  
                                if( dr < dr_gp_leadphoton ) {
                                    dr_gp_leadphoton = dr;
                                    index_gp_leadphoton = genLoop;
                                }
                
                                dr = deltaR( truth_obj.diPhoton()->subLeadingPhoton()->eta(), truth_obj.diPhoton()->subLeadingPhoton()->phi(), part->eta(), part->phi() );
                 
                                if( dr < dr_gp_subleadphoton ) {
                                    dr_gp_subleadphoton = dr;
                                    index_gp_subleadphoton = genLoop;
                                }
                            }
        
                            if (truth_obj.hasLeadingMuon()) {
                                dr = deltaR( truth_obj.leadingMuon()->eta(), truth_obj.leadingMuon()->phi(), part->eta(), part->phi() );
                                if( dr < dr_gp_leadmuon ) {
                                    dr_gp_leadmuon = dr;
                                    index_gp_leadmuon = genLoop;
                                }
                            }
    
                            if (truth_obj.hasSubLeadingMuon()) {
                                dr = deltaR( truth_obj.subLeadingMuon()->eta(), truth_obj.subLeadingMuon()->phi(), part->eta(), part->phi() );
                                if( dr < dr_gp_subleadmuon ) {
                                    dr_gp_subleadmuon = dr;
                                    index_gp_subleadmuon = genLoop;
                                }
                            }
        
                            if (truth_obj.hasLeadingElectron()) {
                                dr = deltaR( truth_obj.leadingElectron()->eta(), truth_obj.leadingElectron()->phi(), part->eta(), part->phi() );
                                if( dr < dr_gp_leadelectron ) {
                                    dr_gp_leadelectron = dr;
                                    index_gp_leadelectron = genLoop;
                                }
                            }
        
                            if (truth_obj.hasSubLeadingElectron()) {
                                dr = deltaR( truth_obj.subLeadingElectron()->eta(), truth_obj.subLeadingElectron()->phi(), part->eta(), part->phi() );
                               if( dr < dr_gp_subleadelectron ) {
                                   dr_gp_subleadelectron = dr;
                                   index_gp_subleadelectron = genLoop;
                               }
                            }
                        }
                    } // end of Gen-Reco level matching
    
                    if( index_gp_leadjet < std::numeric_limits<unsigned int>::max() ) { 
                        truth_obj.setClosestParticleToLeadingJet( genParticles->ptrAt( index_gp_leadjet ) ); }
                    if( index_gp_subleadjet < std::numeric_limits<unsigned int>::max() ) {
                        truth_obj.setClosestParticleToSubLeadingJet( genParticles->ptrAt( index_gp_subleadjet ) ); }
                    if( index_gp_leadphoton < std::numeric_limits<unsigned int>::max() ) {  
                        truth_obj.setClosestParticleToLeadingPhoton( genParticles->ptrAt( index_gp_leadphoton ) ); }
                    if( index_gp_subleadphoton < std::numeric_limits<unsigned int>::max() ) {
                        truth_obj.setClosestParticleToSubLeadingPhoton( genParticles->ptrAt( index_gp_subleadphoton ) );
                    }
    
                    if( index_gp_leadmuon < std::numeric_limits<unsigned int>::max() ) {
                        truth_obj.setClosestParticleToLeadingMuon( genParticles->ptrAt( index_gp_leadmuon ) );
                        const reco::GenParticle *mcMom;
                        mcMom = static_cast<const reco::GenParticle *>(genParticles->ptrAt( index_gp_leadmuon )->mother());
                        if (mcMom) {
                            if( abs(genParticles->ptrAt( index_gp_leadmuon )->pdgId())==13 
                            && genParticles->ptrAt( index_gp_leadmuon )->status()==1  
                            && abs( mcMom->pdgId())==24 ) {
                                truth_obj.setClosestPromptParticleToLeadingMuon( genParticles->ptrAt( index_gp_leadmuon ) );
                            }
                        }
                    }
        
                if( index_gp_subleadmuon < std::numeric_limits<unsigned int>::max() ) {
                    truth_obj.setClosestParticleToSubLeadingMuon( genParticles->ptrAt( index_gp_subleadmuon ) );
                    const reco::GenParticle *mcMom;
                    mcMom = static_cast<const reco::GenParticle *>(genParticles->ptrAt( index_gp_subleadmuon )->mother());
                    if (mcMom){
                        if( abs(genParticles->ptrAt( index_gp_subleadmuon )->pdgId())==13 
                        && genParticles->ptrAt( index_gp_subleadmuon )->status()==1  
                        && abs( mcMom->pdgId())==24 ) {
                            truth_obj.setClosestPromptParticleToSubLeadingMuon( genParticles->ptrAt( index_gp_subleadmuon ) );
                        }
                    }
                }
    
                if( index_gp_leadelectron < std::numeric_limits<unsigned int>::max() ) {
                    truth_obj.setClosestParticleToLeadingElectron( genParticles->ptrAt( index_gp_leadelectron ) );
                    const reco::GenParticle *mcMom;
                    mcMom = static_cast<const reco::GenParticle *>(genParticles->ptrAt( index_gp_leadelectron )->mother());
                    if (mcMom){
                        if( abs(genParticles->ptrAt( index_gp_leadelectron )->pdgId())==11 
                        && genParticles->ptrAt( index_gp_leadelectron )->status()==1  
                        && abs( mcMom->pdgId())==24 ) {
                            truth_obj.setClosestPromptParticleToLeadingElectron( genParticles->ptrAt( index_gp_leadelectron ) );
                        }
                    }
                }
    
                if( index_gp_subleadelectron < std::numeric_limits<unsigned int>::max() ) {
                    const reco::GenParticle *mcMom;
                    truth_obj.setClosestParticleToSubLeadingElectron( genParticles->ptrAt( index_gp_subleadelectron ) );
                    mcMom = static_cast<const reco::GenParticle *>(genParticles->ptrAt( index_gp_subleadelectron )->mother());
                    if (mcMom) {
                        if( abs(genParticles->ptrAt( index_gp_subleadelectron )->pdgId())==11 
                        && genParticles->ptrAt( index_gp_subleadelectron )->status()==1  
                        && abs( mcMom->pdgId())==24 ) {
                            truth_obj.setClosestPromptParticleToSubLeadingElectron( genParticles->ptrAt( index_gp_subleadelectron ) );
                        }
                    }
                }

                unsigned int index_gj_leadjet = std::numeric_limits<unsigned int>::max();
                unsigned int index_gj_subleadjet = std::numeric_limits<unsigned int>::max();
                unsigned int index_gj_subsubleadjet = std::numeric_limits<unsigned int>::max();

                float dr_gj_leadjet = 999.;
                float dr_gj_subleadjet = 999.;
                float dr_gj_subsubleadjet = 999.;

                //GEN Jet-RECO Jet Matching
                for( unsigned int gjLoop = 0 ; gjLoop < genJets->size() ; gjLoop++ ) {
                    edm::Ptr <reco::GenJet> gj = genJets->ptrAt( gjLoop );
                   float dr = deltaR( SelJetVect_PtSorted[0]->eta(), SelJetVect_PtSorted[0]->phi(), gj->eta(), gj->phi() );
                    if( dr < dr_gj_leadjet ) {
                        dr_gj_leadjet = dr;
                        index_gj_leadjet = gjLoop;
                    }
                    //if(  > 1 ){
                    dr = deltaR( SelJetVect_PtSorted[1]->eta(), SelJetVect_PtSorted[1]->phi(), gj->eta(), gj->phi() );
                    if( dr < dr_gj_subleadjet ) {
                        dr_gj_subleadjet = dr;
                        index_gj_subleadjet = gjLoop;
                    }
                    //}
                    if (truth_obj.hasSubSubLeadingJet()) {
                        dr = deltaR( SelJetVect_PtSorted[2]->eta(), SelJetVect_PtSorted[2]->phi(), gj->eta(), gj->phi() );
                        if( dr < dr_gj_subsubleadjet ) {
                            dr_gj_subsubleadjet = dr;
                            index_gj_subsubleadjet = gjLoop;
                        }
                    }
                }
    
                if( index_gj_leadjet < std::numeric_limits<unsigned int>::max() ) {  truth_obj.setClosestGenJetToLeadingJet( genJets->ptrAt( index_gj_leadjet ) ); }
                if( index_gj_subleadjet < std::numeric_limits<unsigned int>::max() ) { truth_obj.setClosestGenJetToSubLeadingJet( genJets->ptrAt( index_gj_subleadjet ) ); }
                if( index_gj_subsubleadjet < std::numeric_limits<unsigned int>::max() ) { truth_obj.setClosestGenJetToSubSubLeadingJet( genJets->ptrAt( index_gj_subsubleadjet ) ); }
          
                //Parton-Jet Matching
                std::vector<edm::Ptr<reco::GenParticle>> ptOrderedPartons;
                for (unsigned int genLoop(0);genLoop < genParticles->size();genLoop++) {
                    edm::Ptr<reco::GenParticle> gp = genParticles->ptrAt(genLoop);
                    bool isQuark = abs( gp->pdgId() ) < 7 && gp->numberOfMothers() == 0;
                    bool isGluon = gp->pdgId() == 21 && gp->numberOfMothers() == 0;
                    if (isGluon || isQuark) {
                        unsigned int insertionIndex(0);
                        for (unsigned int parLoop(0);parLoop<ptOrderedPartons.size();parLoop++) {
                            if (gp->pt() < ptOrderedPartons[parLoop]->pt()) { insertionIndex = parLoop + 1; }
                        }
                        ptOrderedPartons.insert( ptOrderedPartons.begin() + insertionIndex, gp);
                    }
                }
    
                //Lead
                if ( ptOrderedPartons.size() > 0 && truth_obj.hasLeadingJet()) {
                    float dr(999.0);
                    unsigned pIndex(0);
                    for (unsigned partLoop(0);partLoop<ptOrderedPartons.size();partLoop++) {
                        float deltaR_temp = deltaR(SelJetVect_PtSorted[0]->eta(),SelJetVect_PtSorted[0]->phi(),
                        ptOrderedPartons[partLoop]->eta(),ptOrderedPartons[partLoop]->phi());
                        if (deltaR_temp < dr) {dr = deltaR_temp; pIndex = partLoop;}
                    }
                    truth_obj.setClosestPartonToLeadingJet( ptOrderedPartons[pIndex] );
                }

                //Sublead
                if (ptOrderedPartons.size() > 0 && truth_obj.hasSubLeadingJet()) {
                    float dr(999.0);
                    unsigned pIndex(0);
                    for (unsigned partLoop(0);partLoop<ptOrderedPartons.size();partLoop++) {
                        float deltaR_temp = deltaR(SelJetVect_PtSorted[1]->eta(),SelJetVect_PtSorted[1]->phi(),
                        ptOrderedPartons[partLoop]->eta(),ptOrderedPartons[partLoop]->phi());
                        if (deltaR_temp < dr) {dr = deltaR_temp; pIndex = partLoop;}
                    }
                    truth_obj.setClosestPartonToSubLeadingJet( ptOrderedPartons[pIndex] );
                }

                //Subsublead
                if (ptOrderedPartons.size() > 0 && truth_obj.hasSubSubLeadingJet()) {
                    float dr(999.0);
                    unsigned pIndex(0);
                    for (unsigned partLoop(0);partLoop<ptOrderedPartons.size();partLoop++) {
                        float deltaR_temp = deltaR(SelJetVect_PtSorted[2]->eta(),SelJetVect_PtSorted[2]->phi(),
                                           ptOrderedPartons[partLoop]->eta(),ptOrderedPartons[partLoop]->phi());
                        if (deltaR_temp < dr) {dr = deltaR_temp; pIndex = partLoop;}
                    }
                    truth_obj.setClosestPartonToSubSubLeadingJet( ptOrderedPartons[pIndex] );
                }

                //std::cout<< index_gp_leadmuon << " "<< index_gp_leadjet <<" "<< index_gp_subleadjet <<" "<< index_gp_leadphoton << index_gp_subleadphoton <<" "<<index_gj_leadjet <<  " "<< index_gj_subleadjet << " "<< dr_gp_leadjet << " "<<dr_gp_subleadjet << " "<<dr_gp_leadphoton << " "<<dr_gp_subleadphoton<<" "<< dr_gj_leadjet <<" "<< dr_gj_subleadjet<<std::endl;
            }
        
            thqltags->push_back( thqltags_obj );
            truths->push_back( truth_obj );
            thqltags->back().setTagTruth( edm::refToPtr( edm::Ref<vector<THQLeptonicTagTruth> >( rTagTruth, idx++ ) ) );

        } else{

          thqltags->push_back( thqltags_obj );
        } //FIXME at next iteration!!
    
        } else { // photon selection
                    
          //if(false)
          std::cout << " THQLeptonicTagProducer NO TAG " << std::endl;
        }
      
    } //diPho loop end !
    
    
    particles_LorentzVector.clear();
    particles_RhoEtaPhiVector.clear();
    SelJetVect.clear(); SelJetVect_EtaSorted.clear(); SelJetVect_PtSorted.clear(); SelJetVect_BSorted.clear();        
    LooseBJetVect.clear(); LooseBJetVect_PtSorted.clear(); 
    MediumBJetVect.clear(); MediumBJetVect_PtSorted.clear();
    TightBJetVect.clear(); TightBJetVect_PtSorted.clear();        
    evt.put(std::move( thqltags ) );
    evt.put(std::move( truths ) );
  }
   
}

/* ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---
 * ---- Function to calculate neutrino Pz component ----
 * ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- --- */

int neutrino_reco(TLorentzVector leptonLorentz, TLorentzVector missingEnergyLorentz,
                  TLorentzVector & neutrino_solution_1, TLorentzVector & neutrino_solution_2){
    
  int numb_solutions = 0;    

  // 3-vectors
  TVector3 lepton_pT(leptonLorentz.Px(), leptonLorentz.Py(), leptonLorentz.Pz());
  TVector3 neutrino_pT(missingEnergyLorentz.Px(), missingEnergyLorentz.Py(), missingEnergyLorentz.Pz());
                             
  // Set Z component to zero
  lepton_pT.SetZ(0);
  neutrino_pT.SetZ(0);
                                                         
  float mu = mass_w * mass_w / 2 + lepton_pT * neutrino_pT;
  float A = - (lepton_pT * lepton_pT);
  float B = mu * leptonLorentz.Pz();
  float C = mu * mu - leptonLorentz.E() * leptonLorentz.E() * (neutrino_pT * neutrino_pT);
  float discriminant = B * B - A * C;
                                                                                            
  if(0 >= discriminant){
                                                                                                                    
    // Take only real part of the solution for pz:
    neutrino_solution_1.SetPxPyPzE(missingEnergyLorentz.Px(), missingEnergyLorentz.Py(), -B / A, 0.0);
    neutrino_solution_1.SetE(neutrino_solution_1.P());
                                                                                                                                                           
    numb_solutions = 1;
 
  } else{
                                                                                                                           
    discriminant = sqrt(discriminant);
 
    // Solution 1
    neutrino_solution_1.SetPxPyPzE(missingEnergyLorentz.Px(), missingEnergyLorentz.Py(), (-B - discriminant) / A, 0.0);
    neutrino_solution_1.SetE(neutrino_solution_1.P());

    // Solution 2
    neutrino_solution_2.SetPxPyPzE(missingEnergyLorentz.Px(), missingEnergyLorentz.Py(), (-B + discriminant) / A, 0);
    neutrino_solution_2.SetE(neutrino_solution_2.P());
                    
    numb_solutions = 2;

  }

  return numb_solutions;
}

typedef flashgg::THQLeptonicTagProducer FlashggTHQLeptonicTagProducer;
DEFINE_FWK_MODULE( FlashggTHQLeptonicTagProducer );
