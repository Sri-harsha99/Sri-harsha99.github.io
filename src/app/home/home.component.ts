import { Component, OnInit, Input } from '@angular/core';
import {
  HttpBackend,
  HttpClient,
  HttpHeaders,
  HttpParams,
} from "@angular/common/http";
import { HomeService } from '../home.service';
import * as tf from '@tensorflow/tfjs';
import words from '../../assets/word_dict.json';
import { ChartOptions } from 'chart.js';
import { takeUntil } from 'rxjs/operators';
import { Subject } from 'rxjs';
import { any } from '@tensorflow/tfjs';
import { faBriefcase } from '@fortawesome/free-solid-svg-icons';
import { faGraduationCap } from '@fortawesome/free-solid-svg-icons';
import { faCode } from '@fortawesome/free-solid-svg-icons';
import { faDatabase } from '@fortawesome/free-solid-svg-icons';
import { faDownload } from '@fortawesome/free-solid-svg-icons';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {

  inputReddit = "";
  from = "";
  to = "";
  data:any;
  fileUrl = 'assets/VenkataSriHarshaMaddiralaResume.pdf';
  fileName = 'Venkata Sri Harsha Resume.pdf'
  fromDate:any;
  toDate:any;
  done = false;
  posNumber: any;
  negNumber: any;
  tokenizer:any;
  panelOpenState = false;
  profile:any;
  faBriefcase = faBriefcase;
  faGraduationCap = faGraduationCap;
  faDatabase = faDatabase;
  faDownload = faDownload;
  isCall = false;
  wordDict: any;
  maximum = 500;
  currPositiveTweets:any = [];
  currNegativeTweets:any = [];
  positiveTweets:any = [];
  negativeTweets:any = [];
  countError = false;
  @Input() type!: string;
  nodeURL = 'https://serverless-tweet.azurewebsites.net/api/HttpTrigger3?code=7J111qj4uroIibIKhVlrOnXLA6IWqu2kPCj7IGL_azqMAzFunK1Y2Q==';
  public pieChartOptions: ChartOptions<'pie'> = {
    responsive: false,
  };
  public pieChartLabels = [ [ 'Negative'], [ 'Positive']];
  public pieChartDatasets = [ {
    data: [ 300, 500]
  } ];
  destroy$: Subject<boolean> = new Subject<boolean>();
  public pieChartLegend = true;
  fromFilter = (d: Date | null): boolean => {
    const day = (d || new Date()).getDay();
    // Prevent Saturday and Sunday from being selected.
    return day !== 0 && day !== 6;
  };
  toFilter = (d: Date | null): boolean => {
    const day = (d || new Date()).getDay();
    // Prevent Saturday and Sunday from being selected.
    return day !== 0 && day !== 6;
  };
  minDate = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
  maxDate = new Date();
  toMinDate = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);

  constructor(    private http: HttpClient,private homeService: HomeService) { }

  ngOnInit(): void {  
  }

  getData(){
    if(this.maximum > 5000){
      this.maximum = 5000;
    }
    if(!this.inputReddit.length){
      return
    }
    this.done = false;
    this.isCall = true;
    let pos = 0;
    let neg = 0;
    this.emptyData();
    this.homeService.homeApi({ query: this.inputReddit, startTime:this.fromDate, endTime: this.toDate, max:this.maximum }).pipe(takeUntil(this.destroy$)).subscribe((data: any) => {
      if(data){
        console.log(data);
        this.done = true;
        let predicts = data.predicts.slice(0, -1);
        predicts = predicts.substring(1);
        predicts = predicts.split(" ");
        predicts.forEach((ele: any,index: any)=> {
            if(ele == '1'){
              pos += 1
              this.positiveTweets.push(data.tweets[index]);
            }else{data
              neg += 1
              this.negativeTweets.push(data.tweets[index]);
            }
        });
        this.pieChartDatasets = [ { data: [ neg, pos]} ];
        this.currPositiveTweets = this.positiveTweets.slice(0,100)
        this.currNegativeTweets = this.negativeTweets.slice(0,100)
        this.posNumber = this.positiveTweets.length;
        this.negNumber = this.negativeTweets.length;
      }
      this.isCall = false;
    });
}

emptyData(){
  this.positiveTweets = [];
  this.negativeTweets = [];
}

maxChanged(data: any){
  console.log(this.fromDate)
  if(data<=5000){
    this.countError = false;
  } else{
    this.countError = true;
  }
}
downloadResume(){

      const a = document.createElement("a");
      a.style.display = "none";
      a.href = this.fileUrl;
      a.download = this.fileName;
      document.body.appendChild(a);
      a.click();
      a.remove();

}

viewResume() {
  window.open(this.fileUrl, "_blank");
}

  changePage(page:any,type:any){
    if(type === 'positive'){
      this.currPositiveTweets = this.positiveTweets.slice(page.pageIndex*100,(page.pageIndex+1)*100)
    } else{
      this.currNegativeTweets = this.negativeTweets.slice(page.pageIndex*100,(page.pageIndex+1)*100)
    }
  }

  changeView(){
    this.homeService.type = 'home';
    this.type = 'home';
  }

  openLink(link:any){
    window.open(link, "_blank");
  }

}
