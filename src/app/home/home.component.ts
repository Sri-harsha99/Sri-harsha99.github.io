import { Component, OnInit, Input, ElementRef, ViewChild } from '@angular/core';
import {
  HttpBackend,
  HttpClient,
  HttpHeaders,
  HttpParams,
} from "@angular/common/http";
import { HomeService } from '../home.service';
import { takeUntil } from 'rxjs/operators';
import { Subject } from 'rxjs';
import { faBriefcase } from '@fortawesome/free-solid-svg-icons';
import { faGraduationCap } from '@fortawesome/free-solid-svg-icons';
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
  showCursor1 = true;
  @ViewChild('targetSection1') targetSection1: ElementRef;
  @ViewChild('targetSection2') targetSection2: ElementRef;
  @ViewChild('targetSection3') targetSection3: ElementRef;
  to = "";
  data:any;
  fileUrl = 'assets/HarshaMaddiralaResume.pdf';
  fileName = 'HarshaResume.pdf'
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
  textToType1 = "Hello there, I'm Harsha Maddirala";
  textToType2 = "Graduate CS Student | Software Developer";
  aboutMeText = `A long time ago in a galaxy far, far away... A long time ago in a galaxy far, far away...

  In a city known as Dallas, in the state of Texas, a journey of knowledge and ambition began. A young and determined individual set forth on a path of learning. This individual, with a passion for technology and innovation, embarked on a mission to master the realm of Computer Science.
  
  Under the banner of UT Dallas, the seeker of wisdom, armed with perseverance and dedication, joined the hallowed halls of academia. The academic year of 2022 brought new challenges and opportunities, and our hero embraced them with unwavering resolve.
  
  But this journey was not the first of its kind. In a distant land named Bangalore, India, our hero had previously ventured into the realm of professional software engineering. In the vibrant world of a growth startup named ShopConnect, a product was forged— a beacon of e-commerce. Our hero, a vital cog in the development engine, contributed tirelessly to the creation of this digital marketplace.
  
  Days turned into months, and months into years. With each passing moment, experience was gained, friendships formed, and skills honed. The lessons from ShopConnect were etched into the annals of the hero's journey.
  
  Now, with the year 2024 looming on the horizon, the time has come for our protagonist to take the next step. Master's degree is within reach. Armed with newfound knowledge, experience, and aspirations, the hero gazes towards the future. He is looking for your help in his quest towards triumph.`;

  animationDone = false
  maximum = 500;
  currPositiveTweets:any = [];
  currNegativeTweets:any = [];
  positiveTweets:any = [];
  negativeTweets:any = [];
  countError = false;
  @Input() type!: string;
  nodeURL = 'https://serverless-tweet.azurewebsites.net/api/HttpTrigger3?code=7J111qj4uroIibIKhVlrOnXLA6IWqu2kPCj7IGL_azqMAzFunK1Y2Q==';
  public pieChartOptions: any;
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
  animated = false;
  minDate = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
  maxDate = new Date();
  toMinDate = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);

  constructor(    private http: HttpClient,private homeService: HomeService) { }

  ngOnInit(): void {  
    
  this.homeService.run.subscribe(data => {
    this.ngAfterViewInit();
  });
  }
  onScroll() {
    const scrollTop = window.scrollY || document.documentElement.scrollTop;
    const triggerPoint = 300; // Adjust as needed

    if (scrollTop > triggerPoint && !this.animated) {
      this.animated = true;
      const crawlElement = document.querySelector('.crawl') as HTMLElement;
      crawlElement.classList.add('crawl-animated');
    } else if (scrollTop <= triggerPoint && this.animated) {
      this.animated = false;
      const crawlElement = document.querySelector('.crawl') as HTMLElement;
      crawlElement.classList.remove('crawl-animated');
    }
  }

  onAnimationDone(text: string) {
    this.animationDone = true;
    this.showCursor1 = false;
  }
  ngAfterViewInit() {
    const whiteSpot = document.querySelector('.white-spot') as HTMLElement;

    window.addEventListener('scroll', () => {
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
      const windowHeight = window.innerHeight;

      if (scrollTop <= windowHeight) {
        const scale = 1 + (scrollTop / windowHeight) * 20;
        whiteSpot.style.width = `${scale * 100}px`;
        whiteSpot.style.height = `${scale * 100}px`;
      }
    });
  }

  scrollToSection(id): void {
    if (id === 1) {
      this.targetSection1.nativeElement.scrollIntoView({ behavior: 'smooth' });
    }
    if (id === 2) {
      this.targetSection2.nativeElement.scrollIntoView({ behavior: 'smooth' });
    }
    if (id === 3) {
      this.targetSection3.nativeElement.scrollIntoView({ behavior: 'smooth' });
    }
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

  link(type:any){
    if(type == 'linkedin'){
      window.open('https://www.linkedin.com/in/sriharshamaddirala/', "_blank");
    }else{
      window.open('https://github.com/Sri-harsha99', "_blank");
    }
  }

}
