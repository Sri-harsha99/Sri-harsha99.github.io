import { Component } from '@angular/core';
import { HomeService } from './home.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {

  constructor(public homeService: HomeService) { }

  title = 'find-user';
  
  
  link(type:any){
    if(type == 'linkedin'){
      window.open('https://www.linkedin.com/in/sriharshamaddirala/', "_blank");
    }else{
      window.open('https://github.com/Sri-harsha99', "_blank");
    }
  }
}